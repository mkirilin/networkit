/*
 * ParallelRIGTest.cpp
 *
 *  Created on: Aug 24, 2026
 *      Author: Mikhail Kirilin
 */

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <gtest/gtest.h>

#include <networkit/GlobalState.hpp>
#include <networkit/Globals.hpp>
#include <networkit/auxiliary/Parallelism.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/graph/Graph.hpp>
#include <networkit/io/METISGraphReader.hpp>
#include <networkit/isomorphism/ParallelRI.hpp>
#include <networkit/isomorphism/RI.hpp>
#include <networkit/isomorphism/SubgraphIsomorphism.hpp>

#include "SubgraphIsomorphismTestUtils.hpp"

namespace NetworKit {

using IsomorphismTest::completeGraph;
using IsomorphismTest::graphOf;
using IsomorphismTest::Match;
using IsomorphismTest::sortMatches;
using Semantics = SubgraphIsomorphism::Semantics;

namespace {

/// Unlike the corpus, karate gives a search tree lopsided enough to exercise the work stealing.
Graph karate() {
    METISGraphReader reader;
    return reader.read("input/karate.graph");
}

/// In karate, a 5-path occurs 22 064 times.
Graph path(count n) {
    Graph G(n);
    for (node u = 0; u + 1 < n; ++u)
        G.addEdge(u, u + 1);
    return G;
}

std::vector<int> workerCounts() {
    std::vector<int> counts;
    for (int workers = 1; workers <= 2 * omp_get_num_procs(); workers *= 2)
        counts.push_back(workers);
    return counts;
}

/// Runs @a algo and returns its matches in sorted order.
template <typename Algo>
std::vector<Match> sortedMatches(Algo &&algo) {
    algo.run();
    std::vector<Match> matches = algo.getMatches();
    sortMatches(matches);
    return matches;
}

} // namespace

/// A hanging test points to a termination bug: a worker that never notices that the search is over.
class ParallelRIGTest : public testing::TestWithParam<RI::Variant> {

protected:
    void SetUp() override { threadsBefore = Aux::getMaxNumberOfThreads(); }

    /// Restores the process-global state, also after a failing ASSERT_* returned early.
    void TearDown() override {
        GlobalState::setReceivedSIGINT(false);
        Aux::setNumberOfThreads(threadsBefore);
    }

private:
    int threadsBefore = 1;
};

INSTANTIATE_TEST_SUITE_P(Variants, ParallelRIGTest,
                         testing::Values(RI::Variant::RI, RI::Variant::RI_DS));

/// Python passes the variant as a plain integer, so the constructor must reject unknown values.
TEST_P(ParallelRIGTest, testUnknownVariantIsRejected) {
    const Graph pattern = path(3);
    const Graph target = completeGraph(3);

    EXPECT_THROW(ParallelRI algo(pattern, target, static_cast<RI::Variant>(2)),
                 std::invalid_argument);
    EXPECT_THROW(ParallelRI algo(pattern, target, static_cast<RI::Variant>(256)),
                 std::invalid_argument);
    EXPECT_NO_THROW(ParallelRI algo(pattern, target, GetParam()));
}

TEST_P(ParallelRIGTest, testAgreesWithTheReference) {
    const RI::Variant variant = GetParam();
    const auto make = [variant](const Graph &pattern, const Graph &target, Semantics semantics,
                                count maxMatches) -> std::unique_ptr<SubgraphIsomorphism> {
        return std::make_unique<ParallelRI>(pattern, target, variant, semantics, maxMatches);
    };

    IsomorphismTest::expectMatchesReference(make);
    IsomorphismTest::expectRespectsMatchCap(make);
}

/// Compares the matches element by element, since in a count a lost and a duplicated match would
/// cancel out.
TEST_P(ParallelRIGTest, testAnswerDoesNotDependOnWorkerCount) {
    const Graph target = karate();
    const Graph pattern = path(5);

    const std::vector<Match> expected =
        sortedMatches(RI(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0));
    ASSERT_EQ(expected.size(), 22064u) << "a change here would quietly weaken every case below";

    for (const int workers : workerCounts()) {
        Aux::setNumberOfThreads(workers);

        ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0);
        ASSERT_EQ(algo.numberOfWorkers(), static_cast<count>(workers))
            << "numberOfWorkers() has to follow Aux::setNumberOfThreads()";
        EXPECT_EQ(sortedMatches(algo), expected) << "workers: " << workers;
        EXPECT_EQ(algo.numberOfMatches(), expected.size()) << "workers: " << workers;
    }
}

/// The workers publish their counts in batches and may overshoot the cap, which the stored matches
/// must not show.
TEST_P(ParallelRIGTest, testCapHoldsAtEveryWorkerCount) {
    constexpr count cap = 1000;
    const Graph target = karate();
    const Graph pattern = path(5);

    const std::vector<Match> all =
        sortedMatches(RI(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0));
    ASSERT_GT(all.size(), cap);

    for (const int workers : workerCounts()) {
        Aux::setNumberOfThreads(workers);

        ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, cap);
        const std::vector<Match> actual = sortedMatches(algo);

        EXPECT_EQ(algo.numberOfMatches(), cap) << "workers: " << workers;
        EXPECT_EQ(actual.size(), cap) << "workers: " << workers;
        EXPECT_TRUE(std::includes(all.begin(), all.end(), actual.begin(), actual.end()))
            << "workers: " << workers << " - a stored match is invalid or duplicated";
    }
}

/**
 * A singleton domain drives all three RI-DS rules, and all workers read the domains concurrently.
 * The reference is sequential plain RI, which has no domains and so cannot share a domain bug.
 */
TEST_P(ParallelRIGTest, testSingletonDomainAgreesAtEveryWorkerCount) {
    constexpr node anchor = 0;
    constexpr index rareLabel = 7;

    const Graph target = karate();
    const Graph pattern = path(4);

    // Plain RI would order the pattern endpoint that requires the rare label last.
    std::vector<index> targetNodeLabels(target.upperNodeIdBound(), 0);
    targetNodeLabels[anchor] = rareLabel;
    const std::vector<index> patternNodeLabels{none, none, none, rareLabel};

    RI reference(pattern, target, RI::Variant::RI, Semantics::MONOMORPHISM, 0);
    reference.setNodeLabels(patternNodeLabels, targetNodeLabels);
    const std::vector<Match> expected = sortedMatches(reference);
    ASSERT_FALSE(expected.empty());

    for (const Match &match : expected)
        ASSERT_EQ(match[3], anchor);

    for (const int workers : workerCounts()) {
        Aux::setNumberOfThreads(workers);

        ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0);
        algo.setNodeLabels(patternNodeLabels, targetNodeLabels);
        EXPECT_EQ(sortedMatches(algo), expected) << "workers: " << workers;
    }
}

/// Without nested parallelism, OpenMP starts a single thread for a run() inside a parallel region,
/// while numberOfWorkers() still reports the global thread count. The ring of workers must then
/// span only the started thread, or the search would wait forever for the missing ones.
TEST_P(ParallelRIGTest, testFewerThreadsThanWorkers) {
    const Graph target = karate();
    const Graph pattern = path(5);

    const std::vector<Match> expected =
        sortedMatches(RI(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0));

    Aux::setNumberOfThreads(4);
    // OpenMP 2.0, as in MSVC, has no max-active-levels setting, only the older switch.
#ifndef NETWORKIT_OMP2
    const int levelsBefore = omp_get_max_active_levels();
    omp_set_max_active_levels(1);
#else
    const int nestedBefore = omp_get_nested();
    omp_set_nested(0);
#endif // NETWORKIT_OMP2

    count workers = 0;
    std::vector<Match> actual;
    bool threw = false;

#pragma omp parallel num_threads(2)
    {
#pragma omp single
        {
            // An exception must not leave the region.
            try {
                ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0);
                workers = algo.numberOfWorkers();
                algo.run();
                actual = algo.getMatches();
            } catch (...) {
                threw = true;
            }
        }
    }

    // Restored before any assertion, which could return early.
#ifndef NETWORKIT_OMP2
    omp_set_max_active_levels(levelsBefore);
#else
    omp_set_nested(nestedBefore);
#endif // NETWORKIT_OMP2

    ASSERT_FALSE(threw);
    EXPECT_EQ(workers, 4u) << "the case needs more workers than the nested region starts";
    sortMatches(actual);
    EXPECT_EQ(actual, expected);
}

/// The merged per-worker slots form a multiset, so a duplicated or a lost match changes the sorted
/// sequence. Every worker id must lie in [0, numberOfWorkers()).
TEST_P(ParallelRIGTest, testEveryMatchIsReportedExactlyOnce) {
    Aux::setNumberOfThreads(4);

    const Graph target = karate();
    const Graph pattern = completeGraph(3);

    const std::vector<Match> expected =
        sortedMatches(RI(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0));
    ASSERT_EQ(expected.size(), 270u) << "karate's triangle count is what pins this case";

    ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0);
    ASSERT_EQ(algo.numberOfWorkers(), 4u);

    std::vector<std::vector<Match>> perWorker(algo.numberOfWorkers());
    algo.setCallback([&](index tid, const Match &match) {
        ASSERT_LT(tid, perWorker.size()) << "a worker id outside [0, numberOfWorkers()) would make "
                                            "every documented per-worker accumulator unsafe";
        perWorker[tid].push_back(match);
    });
    algo.run();

    std::vector<Match> collected;
    for (const std::vector<Match> &slot : perWorker)
        collected.insert(collected.end(), slot.begin(), slot.end());

    sortMatches(collected);
    EXPECT_EQ(collected, expected);
    EXPECT_EQ(algo.numberOfMatches(), expected.size());
}

/// A position without a parent draws its candidates from the whole target, or from the domain under
/// RI-DS.
TEST_P(ParallelRIGTest, testDisconnectedPatternIsSearchedInParallel) {
    const Graph target = karate();
    const Graph pattern = graphOf(4, {{0, 1}, {2, 3}});

    const std::vector<Match> expected =
        sortedMatches(RI(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0));
    ASSERT_GT(expected.size(), 1000u);

    EXPECT_EQ(sortedMatches(ParallelRI(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0)),
              expected);
}

/// The empty mapping alone reaches a cap of one, so seeding stops the search.
TEST_P(ParallelRIGTest, testEmptyPatternReachesACapOfOne) {
    // SubgraphIsomorphism holds the graphs by pointer, so they must not be temporaries.
    const Graph k4 = completeGraph(4);
    const Graph nothing(0);

    ParallelRI emptyCapped(nothing, k4, GetParam(), Semantics::INDUCED, 1);
    emptyCapped.run();
    ASSERT_TRUE(emptyCapped.hasFinished());
    EXPECT_EQ(emptyCapped.numberOfMatches(), 1u);
}

/// The binary hangs here instead of failing if a worker keeps spinning after CTRL+C.
TEST_P(ParallelRIGTest, testInterruptStopsEveryWorker) {
    const Graph target = karate();
    const Graph pattern = path(5);

    ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0);

    std::atomic<count> delivered{0};
    algo.setCallback([&](index, const Match &) {
        if (delivered.fetch_add(1) + 1 == 5)
            GlobalState::setReceivedSIGINT(true);
    });

    EXPECT_THROW(algo.run(), Aux::SignalHandler::InterruptException);
    GlobalState::setReceivedSIGINT(false);

    EXPECT_FALSE(algo.hasFinished()) << "an interrupted run must not count as finished";
    EXPECT_THROW(algo.numberOfMatches(), std::runtime_error);
    EXPECT_GE(delivered.load(), 5u) << "matches already handed over cannot be taken back";
}

/// With a single worker, as in RI, no match follows the interrupt, not even one that completes the
/// same state as the match before it.
TEST_P(ParallelRIGTest, testSingleWorkerReportsNothingAfterTheInterrupt) {
    const Graph target = karate();
    const Graph pattern = path(5);

    Aux::setNumberOfThreads(1);
    ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0);
    ASSERT_EQ(algo.numberOfWorkers(), 1u);

    count delivered = 0;
    algo.setCallback([&](const Match &) {
        if (++delivered == 5)
            GlobalState::setReceivedSIGINT(true);
    });

    EXPECT_THROW(algo.run(), Aux::SignalHandler::InterruptException);
    GlobalState::setReceivedSIGINT(false);

    EXPECT_EQ(delivered, 5u);
}

/// An exception that escaped the OpenMP region would terminate the binary. The serial callback form
/// must also release its lock, or the next worker would block forever.
TEST_P(ParallelRIGTest, testThrowingCallbackStopsEveryWorker) {
    // Local, so that no other runtime_error out of run() can satisfy EXPECT_THROW.
    struct CallbackFailure : std::runtime_error {
        CallbackFailure() : std::runtime_error("callback failed") {}
    };

    const Graph target = karate();
    const Graph pattern = path(5);
    const count expected =
        sortedMatches(RI(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0)).size();

    for (const bool parallelForm : {true, false}) {
        SCOPED_TRACE(parallelForm ? "ParallelMatchCallback" : "MatchCallback");

        ParallelRI algo(pattern, target, GetParam(), Semantics::MONOMORPHISM, 0);
        std::atomic<count> delivered{0};
        if (parallelForm)
            algo.setCallback([&](index, const Match &) {
                delivered.fetch_add(1);
                throw CallbackFailure();
            });
        else
            algo.setCallback([&](const Match &) {
                delivered.fetch_add(1);
                throw CallbackFailure();
            });

        EXPECT_THROW(algo.run(), CallbackFailure);
        EXPECT_FALSE(algo.hasFinished()) << "a run that threw must not count as finished";

        // A worker that kept searching after its own exception would call back a second time.
        EXPECT_GE(delivered.load(), 1u);
        EXPECT_LE(delivered.load(), algo.numberOfWorkers());

        std::atomic<count> counted{0};
        algo.setCallback([&](index, const Match &) { counted.fetch_add(1); });
        algo.run();
        EXPECT_TRUE(algo.hasFinished());
        EXPECT_EQ(counted.load(), expected);
    }
}

} // namespace NetworKit
