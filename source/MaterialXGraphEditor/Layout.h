//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#ifndef MATERIALX_LAYOUT_H
#define MATERIALX_LAYOUT_H

#include <MaterialXGraphEditor/UiNode.h>

#include <unordered_set>

namespace ed = ax::NodeEditor;
namespace mx = MaterialX;

// Automatic node layout using the Sugiyama layered graph drawing framework
// (https://blog.disy.net/sugiyama-method/).
//
// Layout operates over two frames to allow ImGui to measure node sizes.
class Layout
{
  public:
    // Start an automatic layout of the given nodes.
    void start(const std::vector<UiNodePtr>& nodes);

    // Advance the layout by one frame, returning true when complete.
    bool advance(float fontScale);

    // Return true if a layout is in progress.
    bool isPending() const { return _pending; }

  private:
    // Node role for layer assignment
    enum class NodeRole
    {
        Input,
        Output,
        Interior
    };

    using PosIndexMap = std::unordered_map<int, int>;

    // Phase 1: Layer assignment
    NodeRole classifyNode(UiNodePtr node) const;
    void assignLayers(const std::vector<UiNodePtr>& nodes);
    void assignLayerRecursive(UiNodePtr node, int layer,
                              std::unordered_set<int>& visiting);

    // Phase 2: Crossing reduction
    void minimizeCrossings();
    void reorderLayer(int layer, int adjacentLayer);
    float barycenter(UiNodePtr node, int adjacentLayer,
                     const PosIndexMap& posIndex) const;

    // Phase 3: Coordinate assignment
    void assignCoordinates(float fontScale);
    void assignXCoordinates(float scaledGap);
    void assignYCoordinates(float scaledRowGap);
    float layerHeight(int layer, float gap) const;

    // Helpers for Y-coordinate assignment
    std::vector<float> computeIdealY(const std::vector<UiNodePtr>& nodes,
                                     int layer) const;
    void refineOrdering(std::vector<UiNodePtr>& nodes,
                        std::vector<float>& idealY);
    std::vector<float> compactPositions(const std::vector<UiNodePtr>& nodes,
                                        const std::vector<float>& idealY,
                                        int layer, float gap) const;
    void recenterPositions(const std::vector<UiNodePtr>& nodes,
                           const std::vector<float>& idealY,
                           std::vector<float>& positions) const;

  private:
    std::vector<UiNodePtr> _nodes;
    std::unordered_map<int, int> _nodeLayer;
    std::unordered_map<int, std::vector<UiNodePtr>> _layerNodes;
    int _maxLayer = 0;
    bool _pending = false;
    bool _sizingFrame = false;
};

#endif
