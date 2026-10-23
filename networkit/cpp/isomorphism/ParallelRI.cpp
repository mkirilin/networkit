#include <algorithm>
#include <atomic>
#include <deque>
#include <exception>
#include <functional>
#include <iterator>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <omp.h>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/isomorphism/ParallelRI.hpp>

#include "RIImpl.hpp"

namespace NetworKit {

namespace {

using Match = SubgraphIsomorphism::Match;

using IsomorphismDetails::RIImpl;
using IsomorphismDetails::RISearchSetup;

/// Maximum number of matches a worker records between two updates of the shared count.
constexpr count MaxPublishInterval = 64;
/// Shared-count updates per worker to aim for, which bounds the overshoot past maxMatches.
constexpr count PublishRounds = 8;

/// Section 5.2.2 of Kimmig, Meyerhenke and Strash finds 4 best.
constexpr count TaskGroupSize = 4;
constexpr count StealAttempts = 4;

/**
 * The worker pool of @ref ParallelRI. Every worker walks the search tree depth first on its private
 * `Worker::states` and moves its oldest states to `Worker::stealable`. An idle worker reclaims its
 * own published states first and then steals the shallowest states of another worker. A token
 * passed around the ring of workers detects termination; see @ref passToken().
 */
class ParallelRIImpl {

public:
    /// Passes a match to the user's callback. Empty if there is no callback.
    using Deliver = std::function<void(index, const Match &)>;

    /// @a setup must outlive this object.
    ParallelRIImpl(const RISearchSetup &setup, const std::vector<index> &patternNodeLabels,
                   const std::vector<index> &targetNodeLabels,
                   SubgraphIsomorphism::Semantics semantics, Aux::SignalHandler &handler,
                   Deliver deliver, bool storeMatches, count maxMatches, count numWorkers)
        : setup(&setup), deliver(std::move(deliver)), storeMatches(storeMatches),
          maxMatches(maxMatches),
          // Worker is neither copyable nor movable, so the vector is sized on construction.
          workers(numWorkers),
          publishInterval(
              std::clamp<count>(maxMatches / (numWorkers * PublishRounds), 1, MaxPublishInterval)) {
        for (index w = 0; w < workers.size(); ++w) {
            Worker &worker = workers[w];
            worker.batch = worker.untilPublish = publishInterval;
            worker.victimRng.seed(static_cast<std::minstd_rand::result_type>(w + 1));
            // Built here rather than in the parallel region, so that a failure throws normally.
            worker.impl.emplace(setup.patternGraph, setup.targetGraph, patternNodeLabels,
                                targetNodeLabels, setup.ordering, setup.domains, semantics, handler,
                                [this, w](const Match &match) { return recordMatch(w, match); });
        }
    }

    void run() {
        // The early exit of RIImpl::run().
        if (RIImpl::cannotMatch(setup->patternGraph, setup->targetGraph, setup->domains))
            return;

#pragma omp parallel num_threads(static_cast<int>(workers.size()))
        {
            const index tid = static_cast<index>(omp_get_thread_num());

#pragma omp single
            {
                stopOnException([&] { seedRoots(tid); });
            }

            stopOnException([&] { workerLoop(tid); });
        }

        if (failure)
            std::rethrow_exception(failure);
    }

    std::vector<Match> takeMatches() {
        count total = 0;
        for (const Worker &worker : workers)
            total += worker.buffer.size();

        std::vector<Match> merged;
        merged.reserve(total);
        for (Worker &worker : workers)
            merged.insert(merged.end(), std::make_move_iterator(worker.buffer.begin()),
                          std::make_move_iterator(worker.buffer.end()));
        return merged;
    }

    count matchesFound() const {
        count total = 0;
        for (const Worker &worker : workers)
            total += worker.found;
        return total;
    }

private:
    struct alignas(64) Worker {
        /// Optional, since RIImpl cannot be default-constructed.
        std::optional<RIImpl> impl;
        /// Only this worker touches these states.
        std::deque<RIImpl::State> states;
        std::vector<Match> buffer;
        count found = 0;
        /// Matches left in the current batch for the shared count, and the size of that batch.
        count untilPublish = 0;
        count batch = 0;
        /// Picks the victims of this worker's steals. A private generator, so that the steals,
        /// whose number depends on the timing, leave the seeded streams of Aux::Random untouched.
        std::minstd_rand victimRng;

        // Idle workers poll the fields below, so they start a cache line of their own, apart from
        // the fields above that the owner writes on every step.

        /// Guarded by `stealableLock`.
        alignas(64) std::deque<RIImpl::State> stealable;
        /// Size of `stealable`, readable without the lock. Exact while the lock is held.
        std::atomic<count> offered{0};
        std::mutex stealableLock;
    };

    /// The number of threads that OpenMP started, which may be fewer than `workers` holds. The
    /// ring spans only these. Valid inside the parallel region only.
    static count teamSize() { return static_cast<count>(omp_get_num_threads()); }

    void stop() { stopped.store(true, std::memory_order_relaxed); }

    /// Adds the matches to the shared count in batches. A batch never exceeds what was left of the
    /// cap at the previous update, so that a single worker stops exactly at the cap. Returns false
    /// once the search is to stop.
    bool recordMatch(index tid, const Match &match) {
        Worker &worker = workers[tid];
        ++worker.found;

        // prepareRun() stores the matches only if there is no callback.
        if (storeMatches)
            worker.buffer.push_back(match);
        else if (deliver)
            deliver(tid, match);

        if (maxMatches != 0 && --worker.untilPublish == 0) {
            const count total =
                published.fetch_add(worker.batch, std::memory_order_relaxed) + worker.batch;
            if (total >= maxMatches)
                stop();
            else
                worker.batch = worker.untilPublish = std::min(publishInterval, maxMatches - total);
        }

        return !stopped.load(std::memory_order_relaxed);
    }

    /// An exception must not leave an OpenMP region, so this stores the first one for run() to
    /// rethrow after the join, and stops all workers.
    template <typename Step>
    void stopOnException(Step &&step) {
        try {
            step();
        } catch (...) {
            if (!failed.exchange(true))
                failure = std::current_exception();
            stop();
        }
    }

    /// Deals the children of the empty mapping round-robin to the workers. This runs inside
    /// `omp single`, so all other workers wait at its barrier.
    void seedRoots(index tid) {
        RIImpl &impl = *workers[tid].impl;
        std::vector<RIImpl::State> roots;
        if (!impl.expand(impl.rootState(), roots)) {
            stop();
            return;
        }

        const count active = teamSize();
        index next = 0;
        for (RIImpl::State &state : roots) {
            workers[next].states.push_back(std::move(state));
            next = static_cast<index>((next + 1) % active);
        }
    }

    /// Every state is expanded completely, so a position without a parent pushes one state per
    /// target node at once.
    void workerLoop(index tid) {
        Worker &worker = workers[tid];
        RIImpl &impl = *worker.impl;
        RIImpl::State state;
        std::vector<RIImpl::State> children;

        while (!stopped.load(std::memory_order_relaxed)) {
            if (!popLocal(tid, state) && !trySteal(tid, state)) {
                passToken(tid);
                if (tokenHops.load() >= teamSize())
                    return;
                std::this_thread::yield();
                continue;
            }

            children.clear();
            if (!impl.expand(state, children)) {
                stop();
                return;
            }

            for (RIImpl::State &child : children)
                worker.states.push_back(std::move(child));

            // After every state rather than every few pushes, since a state just above complete
            // mappings reports them and pushes nothing.
            coalesceIntoTask(tid);
        }
    }

    bool popLocal(index tid, RIImpl::State &out) {
        Worker &worker = workers[tid];

        if (!worker.states.empty()) {
            out = std::move(worker.states.back());
            worker.states.pop_back();
            return true;
        }

        // Reclaim published states, so that the worker never reports idle while it has work. Only
        // the owner adds to `stealable`, so a zero read here is exact.
        if (worker.offered.load() == 0)
            return false;

        const std::lock_guard<std::mutex> guard(worker.stealableLock);
        if (worker.stealable.empty())
            return false;

        // Take the newest state and leave the oldest ones to thieves.
        out = std::move(worker.stealable.back());
        worker.stealable.pop_back();
        worker.offered.store(worker.stealable.size());
        return true;
    }

    /// Offers the oldest states to thieves once the previous offer is gone.
    void coalesceIntoTask(index tid) {
        Worker &worker = workers[tid];
        if (worker.offered.load() != 0)
            return;

        if (worker.states.size() <= TaskGroupSize)
            return;

        const count batch = std::min<count>(TaskGroupSize, worker.states.size() - TaskGroupSize);

        const std::lock_guard<std::mutex> guard(worker.stealableLock);
        for (count i = 0; i < batch; ++i) {
            worker.stealable.push_back(std::move(worker.states.front()));
            worker.states.pop_front();
        }
        worker.offered.store(worker.stealable.size());
    }

    bool trySteal(index thief, RIImpl::State &out) {
        // pickVictim() needs at least two workers.
        const count active = teamSize();
        if (active < 2)
            return false;

        for (count attempt = 0; attempt < StealAttempts; ++attempt) {
            Worker &victim = workers[pickVictim(thief, active)];

            if (victim.offered.load() == 0)
                continue;

            // A thief never waits for the victim.
            const std::unique_lock<std::mutex> guard(victim.stealableLock, std::try_to_lock);
            if (!guard.owns_lock() || victim.stealable.empty())
                continue;

            // Set while holding the victim's lock, which passToken() relies on.
            tokenDirty.store(true);

            out = std::move(victim.stealable.front());
            victim.stealable.pop_front();
            victim.offered.store(victim.stealable.size());
            return true;
        }

        return false;
    }

    index pickVictim(index thief, count active) {
        // Draw from all workers but one and skip the thief.
        std::uniform_int_distribution<index> pick(0, static_cast<index>(active) - 2);
        const index drawn = pick(workers[thief].victimRng);
        return drawn < thief ? drawn : drawn + 1;
    }

    /**
     * Passes the termination token on if the idle worker @a tid holds it. The search is over once
     * the token has made a full lap without a successful steal.
     *
     * No lap completes while work remains. A worker with work at the end of a lap was idle at its
     * own hop, so it stole the work afterwards. That steal set `tokenDirty` under the victim's lock
     * before a later hop, which voided the lap. The argument needs sequentially consistent
     * operations on `tokenDirty` and `Worker::offered`.
     */
    void passToken(index tid) {
        if (tokenHolder.load() != tid)
            return;

        if (tokenDirty.exchange(false))
            tokenHops.store(0);
        else
            tokenHops.fetch_add(1);

        tokenHolder.store(static_cast<index>((tid + 1) % teamSize()));
    }

    const RISearchSetup *setup;

    Deliver deliver;
    bool storeMatches;
    count maxMatches;

    std::vector<Worker> workers;

    std::atomic<bool> stopped{false};
    std::atomic<bool> failed{false};
    std::exception_ptr failure;
    std::atomic<index> tokenHolder{0};
    /// Set by every successful steal. Voids the current lap of the token.
    std::atomic<bool> tokenDirty{false};
    std::atomic<count> tokenHops{0};

    std::atomic<count> published{0};
    count publishInterval;
};

} // namespace

ParallelRI::ParallelRI(const Graph &pattern, const Graph &target, RI::Variant variant,
                       Semantics semantics, count maxMatches)
    : SubgraphIsomorphism(pattern, target, semantics, maxMatches), variant(variant) {
    // Python passes the variant as an integer, so any value can arrive here.
    if (variant != RI::Variant::RI && variant != RI::Variant::RI_DS)
        throw std::invalid_argument("Unknown RI variant");
}

count ParallelRI::numberOfWorkers() const {
    return static_cast<count>(Aux::getMaxNumberOfThreads());
}

void ParallelRI::run() {
    Aux::SignalHandler handler;

    prepareRun();

    // Read once, so that the search uses the number of workers that numberOfWorkers() reports.
    const count numWorkers = numberOfWorkers();

    const IsomorphismDetails::RISearchSetup setup = IsomorphismDetails::prepareRISearch(
        *pattern, *target, patternNodeLabels, targetNodeLabels, patternEdgeLabels, targetEdgeLabels,
        variant, "ParallelRI");

    // Left empty without a callback, so that no match pays for a call that does nothing.
    ParallelRIImpl::Deliver deliver;
    if (hasCallback())
        deliver = [this](index tid, const Match &match) { invokeCallback(tid, match); };

    ParallelRIImpl impl(setup, patternNodeLabels, targetNodeLabels, semantics, handler,
                        std::move(deliver), storesMatches(), maxMatches, numWorkers);
    impl.run();

    // Throw only after all workers have joined.
    handler.assureRunning();

    finishRun(impl.takeMatches(), impl.matchesFound());
}

} // namespace NetworKit
