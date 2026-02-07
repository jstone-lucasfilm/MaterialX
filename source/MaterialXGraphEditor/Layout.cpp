//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGraphEditor/Layout.h>

#include <cmath>
#include <numeric>

namespace
{

const float COLUMN_GAP = 60.0f;
const float ROW_GAP = 40.0f;
const float MIN_COLUMN_WIDTH = 100.0f;
const float OFFSCREEN_POS = -100000.f;
const int CROSSING_REDUCTION_SWEEPS = 4;

} // anonymous namespace

//
// Phase 1: Layer assignment
//

Layout::NodeRole Layout::classifyNode(UiNodePtr node) const
{
    if (node->getInput())
    {
        return NodeRole::Input;
    }

    if (node->getOutput() || node->getOutputConnections().empty())
    {
        return NodeRole::Output;
    }

    return NodeRole::Interior;
}

void Layout::assignLayerRecursive(UiNodePtr node, int layer,
                                  std::unordered_set<int>& visiting)
{
    int nodeId = node->getId();

    // Guard against cycles in the graph.
    if (visiting.count(nodeId))
    {
        return;
    }

    auto it = _nodeLayer.find(nodeId);
    if (it != _nodeLayer.end() && it->second >= layer)
    {
        return;
    }

    // Remove from old layer if being promoted to a higher one.
    if (it != _nodeLayer.end())
    {
        auto& oldVec = _layerNodes[it->second];
        oldVec.erase(std::remove_if(oldVec.begin(), oldVec.end(),
            [nodeId](const UiNodePtr& n) { return n->getId() == nodeId; }),
            oldVec.end());
    }

    _nodeLayer[nodeId] = layer;
    _layerNodes[layer].push_back(node);
    if (layer > _maxLayer)
    {
        _maxLayer = layer;
    }

    visiting.insert(nodeId);
    for (const UiPinPtr& pin : node->getInputPins())
    {
        UiNodePtr upstream = node->getConnectedNode(pin->getName());
        if (upstream && classifyNode(upstream) != NodeRole::Input)
        {
            assignLayerRecursive(upstream, layer + 1, visiting);
        }
    }
    visiting.erase(nodeId);
}

void Layout::assignLayers(const std::vector<UiNodePtr>& nodes)
{
    _nodeLayer.clear();
    _layerNodes.clear();
    _maxLayer = 0;

    // Traverse upstream from all graph output nodes.
    std::unordered_set<int> visiting;
    for (UiNodePtr node : nodes)
    {
        if (classifyNode(node) == NodeRole::Output)
        {
            assignLayerRecursive(node, 0, visiting);
        }
    }

    // Assign any unreached interior nodes to layer 0.
    for (UiNodePtr node : nodes)
    {
        NodeRole role = classifyNode(node);
        if (role == NodeRole::Interior && _nodeLayer.find(node->getId()) == _nodeLayer.end())
        {
            assignLayerRecursive(node, 0, visiting);
        }
    }

    // Pin graph input nodes to the leftmost layer.
    int inputLayer = _maxLayer + 1;
    for (UiNodePtr node : nodes)
    {
        if (classifyNode(node) == NodeRole::Input)
        {
            _nodeLayer[node->getId()] = inputLayer;
            _layerNodes[inputLayer].push_back(node);
        }
    }
    if (!_layerNodes[inputLayer].empty())
    {
        _maxLayer = inputLayer;
    }
}

//
// Phase 2: Crossing reduction
//

float Layout::barycenter(UiNodePtr node, int adjacentLayer,
                         const PosIndexMap& posIndex) const
{
    float sum = 0.f;
    int count = 0;

    if (adjacentLayer < _nodeLayer.at(node->getId()))
    {
        for (UiNodePtr downNode : node->getOutputConnections())
        {
            auto posIt = posIndex.find(downNode->getId());
            if (posIt != posIndex.end())
            {
                sum += (float) posIt->second;
                count++;
            }
        }
    }

    else if (adjacentLayer > _nodeLayer.at(node->getId()))
    {
        for (const UiPinPtr& pin : node->getInputPins())
        {
            UiNodePtr upstream = node->getConnectedNode(pin->getName());
            if (upstream)
            {
                auto posIt = posIndex.find(upstream->getId());
                if (posIt != posIndex.end())
                {
                    sum += (float) posIt->second;
                    count++;
                }
            }
        }
    }

    return count > 0 ? sum / (float) count : -1.f;
}

void Layout::reorderLayer(int layer, int adjacentLayer)
{
    auto it = _layerNodes.find(layer);
    if (it == _layerNodes.end() || it->second.size() <= 1)
    {
        return;
    }

    auto adjIt = _layerNodes.find(adjacentLayer);
    if (adjIt == _layerNodes.end())
    {
        return;
    }

    // Build position-index map for the adjacent layer.
    PosIndexMap posIndex;
    for (int i = 0; i < (int) adjIt->second.size(); i++)
    {
        posIndex[adjIt->second[i]->getId()] = i;
    }

    // Score each node by its barycenter and sort.
    auto& nodes = it->second;
    std::vector<std::pair<float, UiNodePtr>> scored;
    scored.reserve(nodes.size());
    for (UiNodePtr node : nodes)
    {
        scored.push_back({ barycenter(node, adjacentLayer, posIndex), node });
    }

    std::stable_sort(scored.begin(), scored.end(),
        [](const std::pair<float, UiNodePtr>& a,
           const std::pair<float, UiNodePtr>& b)
        {
            if (a.first < 0.f) return false;
            if (b.first < 0.f) return true;
            return a.first < b.first;
        });

    for (size_t i = 0; i < scored.size(); i++)
    {
        nodes[i] = scored[i].second;
    }
}

void Layout::minimizeCrossings()
{
    for (int sweep = 0; sweep < CROSSING_REDUCTION_SWEEPS; sweep++)
    {
        for (int layer = 1; layer <= _maxLayer; layer++)
        {
            reorderLayer(layer, layer - 1);
        }
        for (int layer = _maxLayer - 1; layer >= 0; layer--)
        {
            reorderLayer(layer, layer + 1);
        }
    }
}

//
// Phase 3: Coordinate assignment
//

float Layout::layerHeight(int layer, float gap) const
{
    auto it = _layerNodes.find(layer);
    if (it == _layerNodes.end())
    {
        return 0.f;
    }
    float total = 0.f;
    for (UiNodePtr node : it->second)
    {
        total += ed::GetNodeSize(node->getId()).y;
    }
    if (it->second.size() > 1)
    {
        total += gap * (float) (it->second.size() - 1);
    }
    return total;
}

void Layout::assignXCoordinates(float scaledGap)
{
    std::unordered_map<int, float> columnWidth;
    for (auto& [layer, nodes] : _layerNodes)
    {
        float maxWidth = MIN_COLUMN_WIDTH;
        for (UiNodePtr node : nodes)
        {
            float w = ed::GetNodeSize(node->getId()).x;
            if (w > maxWidth)
            {
                maxWidth = w;
            }
        }
        columnWidth[layer] = maxWidth;
    }

    std::unordered_map<int, float> layerX;
    layerX[0] = 0.f;
    for (int layer = 1; layer <= _maxLayer; layer++)
    {
        layerX[layer] = layerX[layer - 1] - columnWidth[layer] - scaledGap;
    }

    for (int layer = 0; layer <= _maxLayer; layer++)
    {
        auto it = _layerNodes.find(layer);
        if (it == _layerNodes.end())
        {
            continue;
        }
        float x = layerX[layer];
        for (UiNodePtr node : it->second)
        {
            ed::SetNodePosition(node->getId(), ImVec2(x, 0.f));
            node->setPos(ImVec2(x, 0.f));
        }
    }
}

// Compute the ideal Y-position for each node in a layer, defined as the
// median Y-center of its downstream neighbors in already-positioned layers.
// Returns NaN for nodes with no downstream connections.
std::vector<float> Layout::computeIdealY(const std::vector<UiNodePtr>& nodes,
                                         int layer) const
{
    std::vector<float> idealY(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++)
    {
        std::vector<float> neighborCenters;

        for (UiNodePtr downNode : nodes[i]->getOutputConnections())
        {
            auto layerIt = _nodeLayer.find(downNode->getId());
            if (layerIt != _nodeLayer.end() && layerIt->second < layer)
            {
                ImVec2 downPos = ed::GetNodePosition(downNode->getId());
                ImVec2 downSize = ed::GetNodeSize(downNode->getId());
                neighborCenters.push_back(downPos.y + downSize.y * 0.5f);
            }
        }

        if (!neighborCenters.empty())
        {
            std::sort(neighborCenters.begin(), neighborCenters.end());
            size_t mid = neighborCenters.size() / 2;
            float median;
            if (neighborCenters.size() % 2 == 0)
            {
                median = (neighborCenters[mid - 1] + neighborCenters[mid]) * 0.5f;
            }
            else
            {
                median = neighborCenters[mid];
            }
            float nodeHeight = ed::GetNodeSize(nodes[i]->getId()).y;
            idealY[i] = median - nodeHeight * 0.5f;
        }
        else
        {
            idealY[i] = std::nanf("");
        }
    }
    return idealY;
}

// Refine the within-layer node ordering by sorting on ideal Y-position.
// This complements the Phase 2 barycenter heuristic, which only considers
// adjacent layers, by using actual coordinates to handle long edges that
// span multiple layers.
void Layout::refineOrdering(std::vector<UiNodePtr>& nodes,
                            std::vector<float>& idealY)
{
    std::vector<size_t> order(nodes.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
        [&idealY](size_t a, size_t b)
        {
            if (std::isnan(idealY[a])) return false;
            if (std::isnan(idealY[b])) return true;
            return idealY[a] < idealY[b];
        });

    std::vector<UiNodePtr> sortedNodes(nodes.size());
    std::vector<float> sortedIdealY(nodes.size());
    for (size_t i = 0; i < order.size(); i++)
    {
        sortedNodes[i] = nodes[order[i]];
        sortedIdealY[i] = idealY[order[i]];
    }
    nodes = sortedNodes;
    idealY = sortedIdealY;
}

// Walk top-to-bottom through the ordered nodes, placing each at its ideal
// Y-position or just below the previous node, whichever is lower.
std::vector<float> Layout::compactPositions(const std::vector<UiNodePtr>& nodes,
                                            const std::vector<float>& idealY,
                                            int layer, float gap) const
{
    std::vector<float> positions(nodes.size());
    float currMinY = -layerHeight(layer, gap) * 0.5f;
    for (size_t i = 0; i < nodes.size(); i++)
    {
        float nodeHeight = ed::GetNodeSize(nodes[i]->getId()).y;

        if (!std::isnan(idealY[i]))
        {
            positions[i] = std::max(idealY[i], currMinY);
        }
        else
        {
            positions[i] = currMinY;
        }

        currMinY = positions[i] + nodeHeight + gap;
    }
    return positions;
}

// Shift the compacted positions so that the center of mass of connected
// nodes aligns with their ideal center of mass, removing any directional
// bias from the one-sided compaction.
void Layout::recenterPositions(const std::vector<UiNodePtr>& nodes,
                               const std::vector<float>& idealY,
                               std::vector<float>& positions) const
{
    float placedCenter = 0.f;
    float idealCenter = 0.f;
    int count = 0;
    for (size_t i = 0; i < nodes.size(); i++)
    {
        if (!std::isnan(idealY[i]))
        {
            float h = ed::GetNodeSize(nodes[i]->getId()).y;
            placedCenter += positions[i] + h * 0.5f;
            idealCenter += idealY[i] + h * 0.5f;
            count++;
        }
    }
    if (count > 0)
    {
        placedCenter /= (float) count;
        idealCenter /= (float) count;
        float shift = idealCenter - placedCenter;
        for (size_t i = 0; i < nodes.size(); i++)
        {
            positions[i] += shift;
        }
    }
}

void Layout::assignYCoordinates(float scaledRowGap)
{
    for (int layer = 0; layer <= _maxLayer; layer++)
    {
        auto it = _layerNodes.find(layer);
        if (it == _layerNodes.end() || it->second.empty())
        {
            continue;
        }

        auto& nodes = it->second;

        // Compute ideal positions from downstream neighbors, then refine
        // the node ordering to reduce crossings from long edges.
        std::vector<float> idealY = computeIdealY(nodes, layer);
        refineOrdering(nodes, idealY);

        // Compact positions to prevent overlap, then re-center around the
        // ideal center of mass to remove directional bias.
        std::vector<float> finalY = compactPositions(nodes, idealY, layer, scaledRowGap);
        recenterPositions(nodes, idealY, finalY);

        // Apply final positions to the node editor.
        for (size_t i = 0; i < nodes.size(); i++)
        {
            ImVec2 pos = ed::GetNodePosition(nodes[i]->getId());
            ed::SetNodePosition(nodes[i]->getId(), ImVec2(pos.x, finalY[i]));
            nodes[i]->setPos(ImVec2(pos.x, finalY[i]));
        }
    }
}

void Layout::assignCoordinates(float fontScale)
{
    float scaledGap = COLUMN_GAP * fontScale;
    float scaledRowGap = ROW_GAP * fontScale;

    assignXCoordinates(scaledGap);
    assignYCoordinates(scaledRowGap);
}

//
// Public interface
//

void Layout::start(const std::vector<UiNodePtr>& nodes)
{
    _nodes = nodes;
    _pending = !_nodes.empty();
    _sizingFrame = false;
}

bool Layout::advance(float fontScale)
{
    if (!_pending)
    {
        return true;
    }

    if (!_sizingFrame)
    {
        // Frame 1: assign layers and place nodes for size measurement.
        assignLayers(_nodes);

        float y = OFFSCREEN_POS;
        for (UiNodePtr node : _nodes)
        {
            ed::SetNodePosition(node->getId(), ImVec2(OFFSCREEN_POS, y));
            node->setPos(ImVec2(OFFSCREEN_POS, y));
            y += 150.f;
        }

        _sizingFrame = true;
        return false;
    }

    // Frame 2: sizes are now available.
    minimizeCrossings();
    assignCoordinates(fontScale);

    _nodes.clear();
    _nodeLayer.clear();
    _layerNodes.clear();
    _maxLayer = 0;
    _sizingFrame = false;
    _pending = false;

    return true;
}
