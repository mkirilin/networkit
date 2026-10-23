#ifndef NETWORKIT_CPP_ISOMORPHISM_RI_IMPL_HPP_
#define NETWORKIT_CPP_ISOMORPHISM_RI_IMPL_HPP_

#include <string>
#include <vector>

#include <networkit/Globals.hpp>
#include <networkit/auxiliary/SignalHandling.hpp>
#include <networkit/isomorphism/RI.hpp>
#include <networkit/isomorphism/SubgraphIsomorphism.hpp>

#include "MatchReporter.hpp"
#include "SearchGraph.hpp"

namespace NetworKit {
namespace IsomorphismDetails {

/**
 * The RI search behind @ref RI. @ref run() enumerates all matches depth first.
 */
class RIImpl {

public:
    /// `order[i]` is the pattern node mapped at depth i. `parent[i]` is the position of an earlier
    /// neighbour of `order[i]`, or @ref none if position i starts a new connected component.
    struct Ordering {
        std::vector<node> order;
        std::vector<index> parent;
    };

    /// RI-DS candidate domains, indexed by pattern node. Empty under plain RI.
    struct Domains {
        /// Sorted target nodes that each pattern node can be mapped to.
        std::vector<std::vector<node>> ofPatternNode;

        /// Whether intersecting candidates with the domain pays off. See
        /// MinSweepYieldForSliceIntersection in RIImpl.cpp.
        std::vector<bool> earnsItsKeep;

        bool anyEmpty = false;
    };

    /// Computes the RI-DS domains from degrees and labels, refines them with one pass over the
    /// pattern arcs, and applies forward checking. Returns empty domains under plain RI.
    static Domains computeDomains(const SearchGraph &pattern, const SearchGraph &target,
                                  const std::vector<index> &patternNodeLabels,
                                  const std::vector<index> &targetNodeLabels, RI::Variant variant);

    /// A node of the search tree. `mapping[i]` is the image of `order[i]` for positions below
    /// `depth`.
    struct State {
        std::vector<node> mapping;
        count depth = 0;
    };

    /// Computes the matching order of Bonnici et al. Under RI-DS, single-element domains come
    /// first, and a full tie goes to the smaller domain.
    static Ordering computeOrdering(const SearchGraph &pattern, const Domains &domains);

    /// Whether the sizes, the degrees or an empty RI-DS domain rule out every match. False for an
    /// empty pattern, whose one match is the empty mapping.
    static bool cannotMatch(const SearchGraph &pattern, const SearchGraph &target,
                            const Domains &domains);

    /// @a ordering and @a domains must outlive this object.
    RIImpl(const SearchGraph &pattern, const SearchGraph &target,
           const std::vector<index> &patternNodeLabels, const std::vector<index> &targetNodeLabels,
           const Ordering &ordering, const Domains &domains,
           SubgraphIsomorphism::Semantics semantics, Aux::SignalHandler &handler,
           MatchReporter report);

    State rootState() const;

    void run();

private:
    /// The depth-first search. It keeps one level of candidates per open depth on the heap, so a
    /// large pattern cannot overflow the call stack. It returns early on an interrupt or once the
    /// cap on the number of matches is reached.
    void search();

    /// Appends the candidates for position `state.depth` to @a out. It does not filter out mapped
    /// target nodes; ruleEdgesToPrefix() rejects them.
    void candidatesFor(const State &state, std::vector<node> &out) const;

    bool consistent(const State &state, node tv) const;

    bool ruleEdgesToPrefix(const State &state, node tv) const;

    bool ruleNonEdgesToPrefix(const State &state, node tv) const;

    bool reportMapping(const State &state);

    const SearchGraph *patternGraph;
    const SearchGraph *targetGraph;

    const std::vector<index> *patternNodeLabels;
    const std::vector<index> *targetNodeLabels;

    const Ordering *ordering;
    const Domains *domains;

    SubgraphIsomorphism::Semantics semantics;

    /// The search only polls isRunning() and returns. The caller throws for the interrupt.
    Aux::SignalHandler *handler;

    MatchReporter report;

    SubgraphIsomorphism::Match matchBuffer;
};

struct RISearchSetup {
    SearchGraph patternGraph;
    SearchGraph targetGraph;
    RIImpl::Domains domains;
    RIImpl::Ordering ordering;
};

/// Builds the snapshots, the RI-DS domains and the matching order. Throws `std::runtime_error` if a
/// graph has parallel edges with different edge labels.
RISearchSetup prepareRISearch(const Graph &pattern, const Graph &target,
                              const std::vector<index> &patternNodeLabels,
                              const std::vector<index> &targetNodeLabels,
                              const std::vector<index> &patternEdgeLabels,
                              const std::vector<index> &targetEdgeLabels, RI::Variant variant,
                              const std::string &algorithmName);

} // namespace IsomorphismDetails
} // namespace NetworKit

#endif // NETWORKIT_CPP_ISOMORPHISM_RI_IMPL_HPP_
