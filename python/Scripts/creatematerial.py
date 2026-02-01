#!/usr/bin/env python
'''
Construct a MaterialX file from the textures in the given folder, using the standard data libraries
to build a mapping from texture filenames to shader inputs.

By default the standard_surface shading model is assumed, with the --shadingModel option used to
select any other shading model in the data libraries.
'''

import argparse
import os
import re
from dataclasses import dataclass, field

import MaterialX as mx


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

TEXTURE_EXTENSIONS = frozenset({'exr', 'png', 'jpg', 'jpeg', 'tif', 'hdr'})

# Terms that should be considered equivalent when matching textures to inputs.
# Each group contains terms that can match each other (after normalization).
EQUIVALENCE_GROUPS = [
    frozenset({'basecolor', 'diffuse', 'albedo'}),
    frozenset({'metallic', 'metalness'}),
    frozenset({'occlusion', 'ao', 'ambientocclusion'}),
    frozenset({'emissive', 'emission'}),
]

# Build a lookup from term to its equivalence group for fast access
TERM_TO_GROUP = {term: group for group in EQUIVALENCE_GROUPS for term in group}

# Primary qualifiers that indicate an input is the "main" version of an attribute.
# When multiple inputs match (e.g., specular_roughness, coat_roughness), prefer
# those starting with these qualifiers.
PRIMARY_QUALIFIERS = ('base', 'specular', '')

# Match scores for texture-to-input matching, ordered by priority.
# These values determine which shader input is selected when multiple candidates exist.
class MatchScore:
    EXACT = 1.0              # Normalized names match exactly
    EQUIVALENT_TERM = 0.95   # Terms in same equivalence group (e.g., metallic/metalness)
    PRIMARY_SUFFIX = 0.90    # Input ends with texture name, has primary qualifier prefix
    QUALIFIER_PENALTY = 0.02 # Penalty per position in PRIMARY_QUALIFIERS list
    EQUIVALENT_SUFFIX = 0.85 # Input ends with an equivalent term
    OTHER_SUFFIX = 0.80      # Input ends with texture name, non-primary prefix
    NONE = 0.0               # No match


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class MaterialBuildContext:
    '''Configuration and context for material creation.'''
    shadingModelNodeDef: object  # mx.NodeDef
    outputDirPath: str
    materialName: str
    colorspace: str = 'srgb_texture'
    useTiledImage: bool = False

@dataclass
class ShaderMapping:
    '''Mapping between a texture and a shader input.'''
    inputName: str
    inputType: str
    requiresNormalMap: bool = False

@dataclass
class MaterialBuildState:
    '''Mutable state during material document construction.'''
    doc: object  # mx.Document
    nodeGraph: object  # mx.NodeGraph
    shaderNode: object  # mx.Node
    processedInputs: set = field(default_factory=set)
    udimNumbers: set = field(default_factory=set)


# -----------------------------------------------------------------------------
# Standard Library Loading
# -----------------------------------------------------------------------------

def loadStandardLibraries(customPaths=None, customLibraries=None):
    '''Load and return the standard MaterialX libraries as a document.'''
    stdlib = mx.createDocument()
    searchPath = mx.getDefaultDataSearchPath()
    libraryFolders = []

    # Add custom search paths
    if customPaths:
        for pathList in customPaths:
            for path in pathList:
                searchPath.append(path)

    # Add custom library folders (before defaults so they take precedence)
    if customLibraries:
        for libraryList in customLibraries:
            for library in libraryList:
                libraryFolders.append(library)

    libraryFolders.extend(mx.getDefaultDataLibraryFolders())
    mx.loadLibraries(libraryFolders, searchPath, stdlib)
    return stdlib

def findShadingModelNodeDef(stdlib, shadingModel):
    '''Find the nodedef for the requested shading model.'''
    matchingNodeDefs = stdlib.getMatchingNodeDefs(shadingModel)
    if not matchingNodeDefs:
        return None

    # Prefer the default version if one exists
    for nodeDef in matchingNodeDefs:
        if nodeDef.getAttribute('isdefaultversion') == 'true':
            return nodeDef

    return matchingNodeDefs[0]


# -----------------------------------------------------------------------------
# ORM/ARM Packed Texture Detection
# -----------------------------------------------------------------------------

# ORM/ARM: Occlusion, Roughness, Metallic packed into R, G, B channels.
# This is the dominant industry convention used by Unreal Engine, Substance Painter,
# and most PBR workflows. ARM is an alternate naming (Ambient instead of Occlusion).
ORM_PATTERN = re.compile(
    r'(?i)occlusion\s*roughness\s*metallic|'
    r'ambient\s*roughness\s*metallic|'
    r'[-_](?:orm|arm)(?:[-_]|\.|$)'
)

def isOrmTexture(textureName):
    '''Detect if a texture name matches the ORM/ARM packed texture pattern.'''
    return bool(ORM_PATTERN.search(textureName))


# -----------------------------------------------------------------------------
# UDIM Handling
# -----------------------------------------------------------------------------

UDIM_PATTERN = re.compile(r'\.\d+\.')

@dataclass
class TextureFile:
    '''Information about a texture file, potentially part of a UDIM set.'''
    path: mx.FilePath
    isUdim: bool = False
    udimFiles: list = field(default_factory=list)

    @staticmethod
    def _isUdimPath(filePath):
        '''Check if a file path contains a UDIM pattern.'''
        return bool(UDIM_PATTERN.search(filePath.getBaseName()))

    @staticmethod
    def _findUdimSiblings(filePath):
        '''Find all UDIM tile files that match the given file's pattern.'''
        textureDir = filePath.getParentPath()
        textureName = filePath.getBaseName()
        textureExtension = filePath.getExtension()

        # Build a regex pattern that matches all UDIM variants of this file
        fullNamePattern = UDIM_PATTERN.sub(
            UDIM_PATTERN.pattern.replace('\\', '\\\\'),
            textureName
        )

        return [
            textureDir / f
            for f in textureDir.getFilesInDirectory(textureExtension)
            if re.search(fullNamePattern, f.asString())
        ]

    @staticmethod
    def _extractUdimNumbers(filePaths):
        '''Extract UDIM numbers from a list of file paths.'''
        numbers = []
        for filePath in filePaths:
            match = UDIM_PATTERN.search(filePath.getBaseName())
            if match:
                number = re.search(r'\d+', match.group())
                if number:
                    numbers.append(number.group())
        return numbers

    @classmethod
    def fromPath(cls, filePath):
        '''Create a TextureFile from a file path, detecting UDIM patterns.'''
        if cls._isUdimPath(filePath):
            return cls(
                path=filePath,
                isUdim=True,
                udimFiles=cls._findUdimSiblings(filePath),
            )
        return cls(path=filePath, isUdim=False, udimFiles=[filePath])

    def asPattern(self):
        '''Return the file path, using UDIM token pattern if applicable.'''
        if self.isUdim:
            textureDir = self.path.getParentPath()
            textureName = self.path.getBaseName()
            patternName = UDIM_PATTERN.sub(f'.{mx.UDIM_TOKEN}.', textureName)
            return (textureDir / mx.FilePath(patternName)).asString()
        return self.path.asString()

    def getName(self):
        '''Return the texture name without extension or UDIM numbers.'''
        baseName = self.path.getBaseName()
        if self.isUdim:
            name = UDIM_PATTERN.split(baseName)[0]
        else:
            name = baseName.rsplit('.', 1)[0]
        return re.sub(r'[^\w\s]+', '_', name)

    def getUdimNumbers(self):
        '''Return the list of UDIM tile numbers.'''
        return self._extractUdimNumbers(self.udimFiles) if self.isUdim else []


# -----------------------------------------------------------------------------
# Texture Discovery
# -----------------------------------------------------------------------------

def discoverTextures(textureDir, texturePrefix=None):
    '''Discover texture files in the given directory, grouping UDIM tiles together.'''
    texturePrefix = texturePrefix or ''
    allTextures = []

    for ext in TEXTURE_EXTENSIONS:
        # Find all textures with this extension matching the prefix
        textures = [
            textureDir / f
            for f in textureDir.getFilesInDirectory(ext)
            if f.asString().lower().startswith(texturePrefix.lower())
        ]

        # Group UDIM tiles together
        while textures:
            textureFile = TextureFile.fromPath(textures[0])
            allTextures.append(textureFile)
            for udimFile in textureFile.udimFiles:
                textures.remove(udimFile)

    return allTextures


# -----------------------------------------------------------------------------
# Shader Input Matching
# -----------------------------------------------------------------------------

def inputRequiresNormalMap(shaderInput):
    '''Determine if a shader input requires normalmap processing.'''
    geomProp = shaderInput.getAttribute('defaultgeomprop')
    return geomProp == 'Nworld'

def normalizeForMatching(name):
    '''Normalize a name for comparison by removing non-alphanumeric characters.'''
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

def extractBaseTextureName(textureName):
    '''Extract the semantic base name from a texture filename.'''
    parts = textureName.lower().rsplit('_')
    baseTexName = parts[-1]

    # Handle special case where 'color' follows another word (e.g., 'base_color')
    if baseTexName == 'color' and len(parts) >= 2:
        baseTexName = parts[-2] + baseTexName

    return baseTexName

def scoreInputMatch(normalizedTexName, normalizedInputName):
    '''Score how well a texture name matches a shader input name (0.0 to 1.0).'''
    # Exact match after normalization (highest priority)
    if normalizedInputName == normalizedTexName:
        return MatchScore.EXACT

    # Equivalence group match (e.g., "metallic" matches "metalness")
    texGroup = TERM_TO_GROUP.get(normalizedTexName)
    if texGroup and normalizedInputName in texGroup:
        return MatchScore.EQUIVALENT_TERM

    # Input ends with texture name (e.g., "specular_roughness" ends with "roughness")
    if normalizedInputName.endswith(normalizedTexName):
        prefix = normalizedInputName[:-len(normalizedTexName)]
        # Boost score for primary qualifiers (specular_, base_, or no prefix)
        for i, qualifier in enumerate(PRIMARY_QUALIFIERS):
            if prefix == qualifier:
                return MatchScore.PRIMARY_SUFFIX - (i * MatchScore.QUALIFIER_PENALTY)
        # Other prefixes get a lower score
        return MatchScore.OTHER_SUFFIX

    # Input ends with an equivalent term (reuse texGroup from above)
    if texGroup:
        for equivalent in texGroup:
            if normalizedInputName.endswith(equivalent):
                return MatchScore.EQUIVALENT_SUFFIX

    # No match
    return MatchScore.NONE

def findBestMatchingInput(shaderInputs, normalizedTarget):
    '''Find the shader input with the highest match score for the normalized target name.'''
    bestScore = MatchScore.NONE
    bestInput = None
    for shaderInput in shaderInputs:
        normalizedInputName = normalizeForMatching(shaderInput.getName())
        score = scoreInputMatch(normalizedTarget, normalizedInputName)
        if score > bestScore:
            bestScore = score
            bestInput = shaderInput
            if score == MatchScore.EXACT:
                break
    return bestInput, bestScore

def findBestShaderInput(textureName, shadingModelNodeDef):
    '''Find the shader input that best matches the given texture name.'''
    # Check if this is a packed texture - if so, skip single-input matching
    if isOrmTexture(textureName):
        return None

    # Extract and normalize the base texture name
    baseTexName = extractBaseTextureName(textureName)
    normalizedTexName = normalizeForMatching(baseTexName)

    # Find best matching input
    bestInput, bestScore = findBestMatchingInput(
        shadingModelNodeDef.getActiveInputs(), normalizedTexName)

    if bestScore == MatchScore.NONE:
        return None

    return ShaderMapping(
        inputName=bestInput.getName(),
        inputType=bestInput.getType(),
        requiresNormalMap=inputRequiresNormalMap(bestInput),
    )

def findInputByName(shadingModelNodeDef, targetName):
    '''Find a shader input by matching against a target name using synonym-aware matching.'''
    normalizedTarget = normalizeForMatching(targetName)
    bestInput, bestScore = findBestMatchingInput(
        shadingModelNodeDef.getActiveInputs(), normalizedTarget)
    return bestInput if bestScore > MatchScore.NONE else None


# -----------------------------------------------------------------------------
# Document Building
# -----------------------------------------------------------------------------

def createImageNode(state, context, textureFile, outputType, nodeName=None):
    '''Create an image node for a texture file.'''
    imageCategory = 'tiledimage' if context.useTiledImage else 'image'
    nodeName = nodeName or state.nodeGraph.createValidChildName(textureFile.getName())
    imageNode = state.nodeGraph.addNode(imageCategory, nodeName, outputType)

    # Set color space for color textures
    if outputType in ('color3', 'color4'):
        imageNode.setColorSpace(context.colorspace)

    # Set file path relative to the output directory
    filePathString = os.path.relpath(textureFile.asPattern(), context.outputDirPath)
    imageNode.setInputValue('file', filePathString, 'filename')

    return imageNode

def createNormalMapNode(state, sourceNode, baseName):
    '''Create a normalmap processing node.'''
    normalMapNode = state.nodeGraph.addNode(
        'normalmap',
        f'{baseName}_normalmap',
        'vector3'
    )
    normalMapNode.setConnectedNode('in', sourceNode)
    return normalMapNode

def createChannelExtractNode(state, sourceNode, channel, baseName, channelName):
    '''Create a node to extract a single channel from a color3 texture.'''
    extractNode = state.nodeGraph.addNode(
        'extract',
        f'{baseName}_{channelName}',
        'float'
    )
    extractNode.setConnectedNode('in', sourceNode)
    extractNode.setInputValue('index', channel, 'integer')
    return extractNode

def connectNodeToShaderInput(state, sourceNode, inputName, inputType):
    '''Create a graph output from a node and connect it to a shader input.'''
    outputNode = state.nodeGraph.addOutput(f'{sourceNode.getName()}_output', inputType)
    outputNode.setConnectedNode(sourceNode)

    shaderInput = state.shaderNode.addInput(inputName)
    shaderInput.setConnectedOutput(outputNode)
    shaderInput.setType(inputType)

def processOrmTexture(state, context, textureFile):
    '''Process an ORM/ARM packed texture by creating channel extraction nodes.'''

    # ORM channel mappings: (channel_index, input_name, channel_label)
    ormChannels = [
        (0, 'occlusion', 'R'),
        (1, 'roughness', 'G'),
        (2, 'metallic', 'B'),
    ]

    # Find which channels have valid shader inputs
    validChannels = []
    for channel, inputName, label in ormChannels:
        shaderInput = findInputByName(context.shadingModelNodeDef, inputName)
        if shaderInput and shaderInput.getType() == 'float':
            validChannels.append((channel, shaderInput.getName(), label))

    if not validChannels:
        print(f'Skipping ORM texture {textureFile.getName()} - no matching inputs found')
        return

    # Create the source image node (color3 to preserve all channels)
    baseName = state.nodeGraph.createValidChildName(textureFile.getName())
    imageNode = createImageNode(state, context, textureFile, 'color3', baseName)
    # Packed textures are typically linear data, not sRGB
    imageNode.removeAttribute('colorspace')

    # Create extract nodes for each valid channel
    for channel, inputName, label in validChannels:
        # Skip if already processed
        if inputName in state.processedInputs:
            continue

        state.processedInputs.add(inputName)

        # Create channel extraction node and connect to shader
        extractNode = createChannelExtractNode(state, imageNode, channel, baseName, label)
        connectNodeToShaderInput(state, extractNode, inputName, 'float')

    # Track UDIM numbers
    if textureFile.isUdim:
        state.udimNumbers.update(textureFile.getUdimNumbers())

def processStandardTexture(state, context, textureFile, mapping):
    '''Process a standard (non-packed) texture.'''
    # Create image node
    imageNode = createImageNode(state, context, textureFile, mapping.inputType)

    # Apply normalmap processing if needed
    outputSourceNode = imageNode
    if mapping.requiresNormalMap:
        outputSourceNode = createNormalMapNode(state, imageNode, imageNode.getName())

    # Connect to shader
    connectNodeToShaderInput(state, outputSourceNode, mapping.inputName, mapping.inputType)

    # Track UDIM numbers
    if textureFile.isUdim:
        state.udimNumbers.update(textureFile.getUdimNumbers())

def buildMaterialDocument(textureFiles, context):
    '''Build a MaterialX document from the given textures and configuration.'''
    # Create content document structure
    doc = mx.createDocument()
    shadingModel = context.shadingModelNodeDef.getNodeString()

    nodeGraph = doc.addNodeGraph(f'NG_{context.materialName}')
    shaderNode = doc.addNode(shadingModel, f'SR_{context.materialName}', 'surfaceshader')
    doc.addMaterialNode(f'M_{context.materialName}', shaderNode)

    # Initialize build state
    state = MaterialBuildState(
        doc=doc,
        nodeGraph=nodeGraph,
        shaderNode=shaderNode,
    )

    # Separate textures into ORM and standard
    ormTextures = []
    standardTextures = []

    for textureFile in textureFiles:
        if isOrmTexture(textureFile.getName()):
            ormTextures.append(textureFile)
        else:
            standardTextures.append(textureFile)

    # Process ORM textures first (they set multiple inputs)
    for textureFile in ormTextures:
        processOrmTexture(state, context, textureFile)

    # Process standard textures
    for textureFile in standardTextures:
        textureName = textureFile.getName()

        # Find matching shader input
        mapping = findBestShaderInput(textureName, context.shadingModelNodeDef)
        if not mapping:
            print(f'Skipping {textureFile.path.getBaseName()} which does not match any {shadingModel} input')
            continue

        # Skip inputs that have already been processed
        if mapping.inputName in state.processedInputs:
            continue

        state.processedInputs.add(mapping.inputName)

        processStandardTexture(state, context, textureFile, mapping)

    # Create UDIM set if needed
    if state.udimNumbers:
        geomInfoName = doc.createValidChildName(f'GI_{context.materialName}')
        geomInfo = doc.addGeomInfo(geomInfoName)
        geomInfo.setGeomPropValue('udimset', list(state.udimNumbers), 'stringarray')

    return doc


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--outputFilename', dest='outputFilename', type=str,
        help='Filename of the output MaterialX document.'
    )
    parser.add_argument(
        '--path', dest='paths', action='append', nargs='+',
        help='An additional absolute search path location (e.g. "/projects/MaterialX").'
    )
    parser.add_argument(
        '--library', dest='libraries', action='append', nargs='+',
        help='An additional relative path to a custom data library folder (e.g. "libraries/custom").'
    )
    parser.add_argument(
        '--shadingModel', dest='shadingModel', type=str, default='standard_surface',
        help='The shading model used in analyzing input textures.'
    )
    parser.add_argument(
        '--colorSpace', dest='colorSpace', type=str,
        help='The colorspace in which input textures should be interpreted, defaulting to srgb_texture.'
    )
    parser.add_argument(
        '--texturePrefix', dest='texturePrefix', type=str,
        help='Filter input textures by the given prefix.'
    )
    parser.add_argument(
        '--tiledImage', dest='tiledImage', action='store_true',
        help='Request tiledimage nodes instead of image nodes.'
    )
    parser.add_argument(
        dest='inputDirectory', nargs='?',
        help='Input folder that will be scanned for textures, defaulting to the current working directory.'
    )

    opts = parser.parse_args()

    # Determine texture directory
    texturePath = mx.FilePath.getCurrentPath()
    if opts.inputDirectory:
        texturePath = mx.FilePath(opts.inputDirectory)
        if not texturePath.isDirectory():
            print(f'Input folder not found: {texturePath}')
            return

    # Determine output file
    mtlxFile = texturePath / mx.FilePath('material.mtlx')
    if opts.outputFilename:
        mtlxFile = mx.FilePath(opts.outputFilename)

    # Discover textures
    textureFiles = discoverTextures(texturePath, texturePrefix=opts.texturePrefix)
    if not textureFiles:
        print('No matching textures found in input folder.')
        return

    # Load standard libraries and find shading model
    shadingModel = opts.shadingModel or 'standard_surface'
    stdlib = loadStandardLibraries(opts.paths, opts.libraries)
    shadingModelNodeDef = findShadingModelNodeDef(stdlib, shadingModel)
    if not shadingModelNodeDef:
        print(f'Shading model {shadingModel} not found in the MaterialX data libraries')
        return

    print(f'Analyzing textures in the {texturePath.asString()} folder for the {shadingModel} shading model.')

    # Build context for material creation
    context = MaterialBuildContext(
        shadingModelNodeDef=shadingModelNodeDef,
        outputDirPath=mtlxFile.getParentPath().asString(),
        materialName=mx.createValidName(mtlxFile.getBaseName().rsplit('.', 1)[0]),
        colorspace=opts.colorSpace or 'srgb_texture',
        useTiledImage=opts.tiledImage,
    )

    # Create the MaterialX document
    doc = buildMaterialDocument(textureFiles, context)

    # Output the document
    if opts.outputFilename:
        if not mtlxFile.getParentPath().exists():
            mtlxFile.getParentPath().createDirectory()
        mx.writeToXmlFile(doc, mtlxFile.asString())
        print(f'Wrote MaterialX document to disk: {mtlxFile.asString()}')
    else:
        print('Generated MaterialX document:')
        print(mx.writeToXmlString(doc))


if __name__ == '__main__':
    main()
