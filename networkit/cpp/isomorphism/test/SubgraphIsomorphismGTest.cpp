/*
 * SubgraphIsomorphismGTest.cpp
 *
 * Tests the base class SubgraphIsomorphism: the run() protocol, the two callback forms, the
 * per-worker accumulation of a parallel search, input validation and interruption. Most tests use
 * two stand-ins that report a precomputed match set: ReferenceSubgraphIsomorphism for the
 * sequential and MultiWorkerReporter for the parallel path.
 */

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <gtest/gtest.h>

#include <networkit/GlobalState.hpp>
#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Parallel.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/isomorphism/SubgraphIsomorphism.hpp>

#include "SubgraphIsomorphismTestUtils.hpp"

namespace NetworKit {

using IsomorphismTest::completeGraph;
using IsomorphismTest::graphOf;
using IsomorphismTest::Match;
using IsomorphismTest::referenceMatches;
using IsomorphismTest::Semantics;

class SubgraphIsomorphismGTest : public testing::Test {
protected:
    /// The interrupt tests set a process-global flag, which a failing ASSERT_* would leave set.
    void TearDown() override { GlobalState::setReceivedSIGINT(false); }
};

namespace {

/// Reports the reference matches from several threads, as a parallel algorithm does. It ignores
/// @a maxMatches, like workers that overshoot the cap.
class MultiWorkerReporter final : public SubgraphIsomorphism {

public:
    MultiWorkerReporter(const Graph &pattern, const Graph &target, Semantics semantics,
                        count numWorkers, count maxMatches = 0)
        : SubgraphIsomorphism(pattern, target, semantics, maxMatches), numWorkers(numWorkers) {}

    count numberOfWorkers() const override { return numWorkers; }

    void run() override {
        const std::vector<Match> all = IsomorphismTest::referenceMatches(
            *pattern, *target, semantics, patternNodeLabels, targetNodeLabels);

        Aux::SignalHandler handler;
        prepareRun();

        struct alignas(64) Slot {
            std::vector<Match> buffer;
            count found = 0;
        };
        std::vector<Slot> slots(numWorkers);
        const bool store = storesMatches();

        // libomp leaks memory when a team grows past twice the CPU count, which fails the leak
        // checks. We therefore request at most that many threads.
        const count threads = std::min(numWorkers, 2 * static_cast<count>(omp_get_num_procs()));

#pragma omp parallel num_threads(static_cast<int>(threads))
        {
            const index tid = static_cast<index>(omp_get_thread_num());

            // Stride by the team size, since OpenMP may start fewer threads than requested.
            const count team = static_cast<count>(omp_get_num_threads());

            Slot &slot = slots[tid];
            for (index i = tid; i < all.size(); i += team) {
                // Non-throwing inside the region; the throw happens once after the join.
                if (!handler.isRunning())
                    break;
                ++slot.found;
                if (!invokeCallback(tid, all[i]) && store)
                    slot.buffer.push_back(all[i]);
            }
        }

        handler.assureRunning();

        std::vector<Match> merged;
        count found = 0;
        for (Slot &slot : slots) {
            found += slot.found;
            for (Match &match : slot.buffer)
                merged.push_back(std::move(match));
        }

        finishRun(std::move(merged), found);
    }

private:
    count numWorkers;
};

} // namespace

TEST_F(SubgraphIsomorphismGTest, testHarnessDrivesAnAlgorithmThroughTheRunProtocol) {

    const auto make = [](const Graph &pattern, const Graph &target, Semantics semantics,
                         count maxMatches) -> std::unique_ptr<SubgraphIsomorphism> {
        return std::make_unique<IsomorphismTest::ReferenceSubgraphIsomorphism>(
            pattern, target, semantics, maxMatches);
    };

    IsomorphismTest::expectMatchesReference(make);
    IsomorphismTest::expectRespectsMatchCap(make);
}

TEST_F(SubgraphIsomorphismGTest, testBothCallbackFormsReceiveEveryMatchOnce) {

    // A 4-path in K8 has 1680 matches, so the four workers compete for the callback.
    const Graph pattern = graphOf(4, {{0, 1}, {1, 2}, {2, 3}});
    const Graph target = completeGraph(8);
    const count numWorkers = 4;

    std::vector<Match> expected = referenceMatches(pattern, target, Semantics::MONOMORPHISM);
    IsomorphismTest::sortMatches(expected);
    ASSERT_EQ(expected.size(), 1680u);

    // The mutex keeps a regression from corrupting `collected` instead of failing the assertion.
    std::atomic<int> inside{0};
    std::atomic<int> maxObserved{0};
    std::mutex collectedMutex;
    std::vector<Match> collected;

    MultiWorkerReporter serial(pattern, target, Semantics::MONOMORPHISM, numWorkers);
    serial.setCallback([&](const Match &match) {
        const int now = inside.fetch_add(1, std::memory_order_acq_rel) + 1;
        Aux::Parallel::atomic_max(maxObserved, now);

        {
            const std::lock_guard<std::mutex> guard(collectedMutex);
            collected.push_back(match);
        }

        inside.fetch_sub(1, std::memory_order_acq_rel);
    });
    serial.run();

    EXPECT_EQ(maxObserved.load(), 1) << "the serial callback form was entered concurrently";

    IsomorphismTest::sortMatches(collected);
    EXPECT_EQ(collected, expected);
    EXPECT_EQ(serial.numberOfMatches(), expected.size());

    std::vector<std::vector<Match>> perWorker(numWorkers);

    MultiWorkerReporter parallel(pattern, target, Semantics::MONOMORPHISM, numWorkers);
    parallel.setCallback([&](index tid, const Match &match) {
        ASSERT_LT(tid, numWorkers);
        perWorker[tid].push_back(match); // no lock needed: each tid owns its slot
    });
    parallel.run();

    std::vector<Match> fromWorkers;
    for (const std::vector<Match> &slot : perWorker)
        fromWorkers.insert(fromWorkers.end(), slot.begin(), slot.end());

    IsomorphismTest::sortMatches(fromWorkers);
    EXPECT_EQ(fromWorkers, expected);
    EXPECT_EQ(parallel.numberOfMatches(), expected.size());
}

TEST_F(SubgraphIsomorphismGTest, testTheAnswerDoesNotDependOnTheWorkerCount) {

    const Graph pattern = graphOf(2, {{0, 1}});

    const Graph cycle = graphOf(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}});

    std::vector<Match> expected = referenceMatches(pattern, cycle, Semantics::MONOMORPHISM);
    IsomorphismTest::sortMatches(expected);

    for (count workers : {count{1}, count{2}, count{4}, count{8}}) {
        MultiWorkerReporter algo(pattern, cycle, Semantics::MONOMORPHISM, workers);
        algo.run();

        std::vector<Match> actual = algo.getMatches();
        IsomorphismTest::sortMatches(actual);

        EXPECT_EQ(actual, expected) << "workers: " << workers;
        EXPECT_EQ(algo.numberOfMatches(), expected.size()) << "workers: " << workers;
    }

    const Graph k6 = completeGraph(6);
    const count expectedCount = referenceMatches(pattern, k6, Semantics::MONOMORPHISM).size();

    for (count workers : {count{1}, count{4}}) {
        MultiWorkerReporter algo(pattern, k6, Semantics::MONOMORPHISM, workers);
        algo.setStoreMatches(false);
        algo.run();

        EXPECT_EQ(algo.numberOfMatches(), expectedCount) << "workers: " << workers;
        EXPECT_TRUE(algo.hasMatch()) << "workers: " << workers;
        EXPECT_THROW(algo.getMatches(), std::runtime_error) << "workers: " << workers;
    }

    const Graph k5 = completeGraph(5);

    IsomorphismTest::ReferenceSubgraphIsomorphism sequential(pattern, k5, Semantics::MONOMORPHISM);
    EXPECT_EQ(sequential.numberOfWorkers(), 1u);
}

TEST_F(SubgraphIsomorphismGTest, testOvershootIsTrimmedUnlessACallbackReceivedIt) {

    const Graph pattern = graphOf(2, {{0, 1}});
    const Graph target = completeGraph(5);
    const count total = referenceMatches(pattern, target, Semantics::MONOMORPHISM).size();
    constexpr count cap = 5;
    constexpr count numWorkers = 4;
    ASSERT_GT(total, cap);

    MultiWorkerReporter stored(pattern, target, Semantics::MONOMORPHISM, numWorkers, cap);
    stored.run();
    EXPECT_EQ(stored.numberOfMatches(), cap);
    EXPECT_EQ(stored.getMatches().size(), cap);

    MultiWorkerReporter counted(pattern, target, Semantics::MONOMORPHISM, numWorkers, cap);
    counted.setStoreMatches(false);
    counted.run();
    EXPECT_EQ(counted.numberOfMatches(), cap);

    count delivered = 0;
    MultiWorkerReporter called(pattern, target, Semantics::MONOMORPHISM, numWorkers, cap);
    called.setCallback([&](const Match &) { ++delivered; });
    called.run();
    EXPECT_EQ(delivered, total);
    EXPECT_EQ(called.numberOfMatches(), total) << "delivered matches cannot be taken back";
    EXPECT_THROW(called.getMatches(), std::runtime_error);
}

TEST_F(SubgraphIsomorphismGTest, testInterruptLeavesTheAlgorithmUnfinishedButUsable) {

    // The test sets the global flag instead of raising a real SIGINT.
    const Graph pattern = graphOf(3, {{0, 1}, {1, 2}});
    const Graph target = completeGraph(6);

    std::vector<Match> expected = referenceMatches(pattern, target, Semantics::MONOMORPHISM);
    IsomorphismTest::sortMatches(expected);

    IsomorphismTest::ReferenceSubgraphIsomorphism algo(pattern, target, Semantics::MONOMORPHISM);

    count delivered = 0;
    algo.setCallback([&](const Match &) {
        if (++delivered == 5)
            GlobalState::setReceivedSIGINT(true);
    });

    EXPECT_THROW(algo.run(), Aux::SignalHandler::InterruptException);
    GlobalState::setReceivedSIGINT(false);

    EXPECT_FALSE(algo.hasFinished()) << "an interrupted run must not count as finished";
    EXPECT_THROW(algo.numberOfMatches(), std::runtime_error);
    EXPECT_THROW(algo.hasMatch(), std::runtime_error);

    EXPECT_GE(delivered, 5u);

    // A second object without a callback, since the recovery check needs the stored matches.
    IsomorphismTest::ReferenceSubgraphIsomorphism recovering(pattern, target,
                                                             Semantics::MONOMORPHISM);

    GlobalState::setReceivedSIGINT(true);
    EXPECT_THROW(recovering.run(), Aux::SignalHandler::InterruptException);
    GlobalState::setReceivedSIGINT(false);
    ASSERT_FALSE(recovering.hasFinished());

    recovering.run();

    ASSERT_TRUE(recovering.hasFinished());
    std::vector<Match> actual = recovering.getMatches();
    IsomorphismTest::sortMatches(actual);
    EXPECT_EQ(actual, expected);
}

TEST_F(SubgraphIsomorphismGTest, testSetEdgeLabelsValidatesItsInput) {

    Graph pattern = graphOf(3, {{0, 1}, {1, 2}});
    Graph target = graphOf(4, {{0, 1}, {1, 2}, {2, 3}});

    IsomorphismTest::ReferenceSubgraphIsomorphism unindexed(pattern, target,
                                                            Semantics::MONOMORPHISM);
    ASSERT_FALSE(pattern.hasEdgeIds());
    EXPECT_THROW(unindexed.setEdgeLabels({1, 2}, {1, 2, 3}), std::runtime_error)
        << "edge labels are indexed by edge id, so a graph without ids has no index space";

    // Two empty vectors clear the labels, even without edge ids.
    EXPECT_NO_THROW(unindexed.setEdgeLabels({}, {}));

    pattern.indexEdges();
    EXPECT_THROW(unindexed.setEdgeLabels({1, 2}, {1, 2, 3}), std::runtime_error)
        << "the target needs edge ids as well";

    target.indexEdges();
    IsomorphismTest::ReferenceSubgraphIsomorphism algo(pattern, target, Semantics::MONOMORPHISM);

    EXPECT_THROW(algo.setEdgeLabels({1}, {1, 2, 3}), std::runtime_error) << "pattern too short";
    EXPECT_THROW(algo.setEdgeLabels({1, 2}, {1, 2}), std::runtime_error) << "target too short";
    EXPECT_NO_THROW(algo.setEdgeLabels({1, 2}, {1, 2, 3}));
    EXPECT_NO_THROW(algo.setEdgeLabels({}, {}));
}

} // namespace NetworKit
