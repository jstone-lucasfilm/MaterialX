#!/usr/bin/env python
'''
Construct a MaterialX file from the textures in the given folder, using the standard data libraries
to build a mapping from texture filenames to shader inputs.

By default the standard_surface shading model is assumed, with the --shadingModel option used to
select any other shading model in the data libraries.  Custom data libraries may be made available
through the --path and --library options.

Packed textures such as ORM/ARM (occlusion, roughness, metallic in the R, G, B channels) and glTF
metallicRoughness maps are recognized by name, and their channels are extracted and connected to
the corresponding shader inputs.
'''

import argparse
import itertools
import os
import re
from dataclasses import dataclass

import MaterialX as mx


# -----------------------------------------------------------------------------
# Texture Discovery
# -----------------------------------------------------------------------------

# Texture extensions in search order.  The order is significant, as it determines which file
# is used when the same texture exists in multiple formats, and the order of nodes in the
# generated document.
TEXTURE_EXTENSIONS = ('exr', 'png', 'jpg', 'jpeg', 'tif', 'hdr')

# UDIM tiles are numbered with four digits between dots, e.g. 'wood_color.1001.png'.
UDIM_PATTERN = re.compile(r'\.(\d{4})\.')
UDIM_TOKEN_STRING = f'.{mx.UDIM_TOKEN}.'

class TextureFile:
    '''A texture file, or a set of UDIM tiles of the same texture.'''

    def __init__(self, filePaths):
        firstPath = filePaths[0]
        baseName = firstPath.getBaseName()

        self.filePaths = filePaths
        self.isUdim = bool(UDIM_PATTERN.search(baseName))

        # The file path, with UDIM numbers replaced by the UDIM token
        patternName = UDIM_PATTERN.sub(UDIM_TOKEN_STRING, baseName)
        self.pattern = (firstPath.getParentPath() / mx.FilePath(patternName)).asString()

        # The UDIM numbers of all tiles
        self.udimNumbers = sorted(match.group(1) for filePath in filePaths
                                  for match in [UDIM_PATTERN.search(filePath.getBaseName())] if match)

        # The texture name without extension or UDIM numbers, as a valid MaterialX name
        if self.isUdim:
            name = UDIM_PATTERN.split(baseName)[0]
        else:
            name = baseName.rsplit('.', 1)[0]
        self.name = re.sub(r'[^\w\s]+', '_', name)

def discoverTextures(textureDir, texturePrefix=None):
    '''Discover texture files in the given directory, grouping UDIM tiles of the same texture.'''
    texturePrefix = (texturePrefix or '').lower()

    # Group file paths by their UDIM pattern, preserving the order of discovery
    tileGroups = {}
    for ext in TEXTURE_EXTENSIONS:
        for fileName in textureDir.getFilesInDirectory(ext):
            if not fileName.asString().lower().startswith(texturePrefix):
                continue
            filePath = textureDir / fileName
            pattern = UDIM_PATTERN.sub(UDIM_TOKEN_STRING, filePath.asString())
            tileGroups.setdefault(pattern, []).append(filePath)

    return [TextureFile(filePaths) for filePaths in tileGroups.values()]


# -----------------------------------------------------------------------------
# Shader Input Matching
# -----------------------------------------------------------------------------

# Pattern splitting a name into words at underscores, digits and camelCase boundaries,
# e.g. 'diffuseColor' into ('diffuse', 'Color') and 'specular_IOR' into ('specular', 'IOR').
WORD_PATTERN = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+')

# Conventional abbreviations, expanded word by word before texture and input names are
# compared, so that e.g. a texture named 'wood_rough' matches 'specular_roughness'.
ABBREVIATIONS = {
    'diff': 'diffuse',
    'spec': 'specular',
    'rough': 'roughness',
    'metal': 'metallic',
    'nrm': 'normal',
    'norm': 'normal',
    'disp': 'displacement',
    'emis': 'emissive',
    'opac': 'opacity',
}

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

# Qualifiers that indicate an input is the "main" version of an attribute, in order of
# preference.  When multiple inputs end with the same term (e.g. specular_roughness and
# coat_roughness), those with these qualifiers are preferred.
PRIMARY_QUALIFIERS = ('base', 'specular', '')

# Suffix shared by color-valued shader inputs, e.g. emission_color or diffuseColor.
COLOR_SUFFIX = 'color'

# Suffixes that an input name may add to a texture term, in order of preference.  The bare term
# and its weight are the scalar controls of a feature (e.g. 'coat' in standard_surface and
# 'coat_weight' in OpenPBR), and are preferred over its color (e.g. 'coat_color').
INPUT_SUFFIXES = ('', 'weight', COLOR_SUFFIX)

# Term used for a texture that is simply named 'color', e.g. wood_color.png.
DEFAULT_COLOR_TERM = 'basecolor'

def normalizeName(name):
    '''
    Normalize a name for comparison by splitting it into words, expanding conventional
    abbreviations, and joining the lowercase words without separators.
    '''
    words = [word.lower() for word in WORD_PATTERN.findall(name)]
    return ''.join(ABBREVIATIONS.get(word, word) for word in words)

def relateInputToTerm(inputName, term):
    '''
    Return a rank describing how a normalized input name relates to a normalized term, with
    lower ranks being closer relations, or None if the two are unrelated.

    In order of preference, the input may be the term with one of the INPUT_SUFFIXES (e.g.
    'roughness', 'coat_weight' for 'coat', or 'emissiveColor' for 'emissive'), the term with a
    primary qualifier (e.g. 'specular_roughness'), or the term with any other qualifier
    (e.g. 'coat_roughness').
    '''
    for i, suffix in enumerate(INPUT_SUFFIXES):
        if inputName == term + suffix:
            return (i, 0)
    if inputName.endswith(term):
        qualifier = inputName[:-len(term)]
        if qualifier in PRIMARY_QUALIFIERS:
            return (len(INPUT_SUFFIXES), PRIMARY_QUALIFIERS.index(qualifier))
        return (len(INPUT_SUFFIXES) + 1, 0)
    return None

def inputRequiresNormalMap(shaderInput):
    '''Determine if a shader input expects a world-space normal, requiring normalmap processing.'''
    return shaderInput.getAttribute('defaultgeomprop') == 'Nworld'

@dataclass
class TextureTerm:
    '''The normalized semantic term of a texture name, used to match it to a shader input.'''
    name: str
    qualifier: str = None     # The term preceding 'color', e.g. 'diffuse' for 'diffuse_color'
    isBareColor: bool = False # True for a texture named only '<asset>_color', e.g. 'wood_color'

def extractTextureTerm(textureName):
    '''
    Extract the normalized semantic term from a texture filename.

    The final underscore-separated component is used, e.g. 'roughness' for 'wood_roughness'.
    When that component is 'color' the preceding component is folded in as a qualifier,
    e.g. 'basecolor' for 'wood_base_color', and a component that itself ends in 'color',
    e.g. 'EmissiveColor', yields the remainder as its qualifier.
    '''
    parts = textureName.rsplit('_')
    name = normalizeName(parts[-1])

    if name == COLOR_SUFFIX and len(parts) >= 2:
        qualifier = normalizeName(parts[-2])
        return TextureTerm(qualifier + name, qualifier, isBareColor=(len(parts) == 2))
    if name.endswith(COLOR_SUFFIX) and len(name) > len(COLOR_SUFFIX):
        return TextureTerm(name, name[:-len(COLOR_SUFFIX)])
    return TextureTerm(name)

class ShaderInputMatcher:
    '''Matches texture names and semantic terms to the inputs of a shading model.'''

    def __init__(self, shadingModelNodeDef):
        # Inputs hidden from the user are infrastructure rather than appearance parameters.
        self._inputs = [(normalizeName(shaderInput.getName()), shaderInput)
                        for shaderInput in shadingModelNodeDef.getActiveInputs()
                        if shaderInput.getAttribute('uivisible') != 'false']

    def matchTerm(self, term):
        '''Return the shader input that best matches a normalized term, or None if there is none.'''
        # Candidate terms are the term itself, followed by its equivalents
        candidates = [(term, 0)]
        candidates.extend((equivalent, 1) for equivalent in TERM_TO_GROUP.get(term, ()) if equivalent != term)

        # Select the input with the lowest (relation, qualifier, candidate) rank, preferring
        # earlier inputs when ranks are equal
        bestRank = None
        bestInput = None
        for inputName, shaderInput in self._inputs:
            for candidate, candidateRank in candidates:
                relation = relateInputToTerm(inputName, candidate)
                if relation is None:
                    continue
                rank = relation + (candidateRank,)
                if bestRank is None or rank < bestRank:
                    bestRank = rank
                    bestInput = shaderInput
        return bestInput

    def matchTexture(self, textureName):
        '''Return the shader input that best matches a texture name, or None if there is none.'''
        term = extractTextureTerm(textureName)
        shaderInput = self.matchTerm(term.name)

        # Resolve color textures through their qualifier when the full name finds no input.  A
        # qualifier that is itself a color term is used directly (e.g. 'emission_color' may
        # connect to 'emissive'), while a texture named only '<asset>_color' (e.g. 'wood_color')
        # is treated as the base color.
        if not shaderInput and term.qualifier:
            qualifierInput = self.matchTerm(term.qualifier)
            if qualifierInput and qualifierInput.isColorType():
                shaderInput = qualifierInput
            elif not qualifierInput and term.isBareColor:
                shaderInput = self.matchTerm(DEFAULT_COLOR_TERM)

        return shaderInput


# -----------------------------------------------------------------------------
# Packed Textures
# -----------------------------------------------------------------------------

# Known channel layouts for packed textures, keyed by a pattern that must match the final
# component of the texture name (after the last underscore).  Each layout lists the
# (channel index, semantic term) pairs that the texture provides.
#
# ORM/ARM: Occlusion, Roughness, Metallic packed into R, G, B channels.  This is the dominant
# convention used by Unreal Engine, Substance Painter, and most PBR workflows.  ARM is an
# alternate naming (Ambient instead of Occlusion).
#
# metallicRoughness: the glTF convention, with roughness in G and metallic in B.
ORM_CHANNELS = ((0, 'occlusion'), (1, 'roughness'), (2, 'metallic'))
METALLIC_ROUGHNESS_CHANNELS = ((1, 'roughness'), (2, 'metallic'))
PACKED_TEXTURE_LAYOUTS = (
    (r'(?:occlusion|ambient)[_ ]?roughness[_ ]?metallic|orm|arm', ORM_CHANNELS),
    (r'metallic[_ ]?roughness', METALLIC_ROUGHNESS_CHANNELS),
)
PACKED_TEXTURE_PATTERNS = [
    (re.compile(rf'(?:^|_)(?:{pattern})$', re.IGNORECASE), channels)
    for pattern, channels in PACKED_TEXTURE_LAYOUTS
]

CHANNEL_LABELS = ('R', 'G', 'B', 'A')

def findPackedLayout(textureName):
    '''Return the channel layout for a packed texture name, or None if the name is not recognized.'''
    for pattern, channels in PACKED_TEXTURE_PATTERNS:
        if pattern.search(textureName):
            return channels
    return None


# -----------------------------------------------------------------------------
# Document Building
# -----------------------------------------------------------------------------

class MaterialBuilder:
    '''Builds a MaterialX document that connects textures to the inputs of a shading model.'''

    def __init__(self, shadingModelNodeDef, materialName, outputDir,
                 colorspace='srgb_rec709_scene', useTiledImage=False):
        self.shadingModel = shadingModelNodeDef.getNodeString()
        self.matcher = ShaderInputMatcher(shadingModelNodeDef)
        self.materialName = materialName
        self.outputDir = outputDir
        self.colorspace = colorspace
        self.imageCategory = 'tiledimage' if useTiledImage else 'image'

        self.doc = mx.createDocument()
        self.nodeGraph = self.doc.addNodeGraph(f'NG_{materialName}')
        self.shaderNode = self.doc.addNode(self.shadingModel, f'SR_{materialName}', 'surfaceshader')
        self.doc.addMaterialNode(f'M_{materialName}', self.shaderNode)

        self._connectedInputs = {}  # Input name -> name of the texture connected to it
        self._udimNumbers = set()

    def build(self, textureFiles):
        '''Connect the given textures to the shading model and return the document.'''
        # Process packed textures first, as they connect multiple inputs
        standardTextures = []
        for textureFile in textureFiles:
            channels = findPackedLayout(textureFile.name)
            if channels:
                self._addPackedTexture(textureFile, channels)
            else:
                standardTextures.append(textureFile)
        for textureFile in standardTextures:
            self._addTexture(textureFile)

        # Create the UDIM set if needed
        if self._udimNumbers:
            geomInfoName = self.doc.createValidChildName(f'GI_{self.materialName}')
            geomInfo = self.doc.addGeomInfo(geomInfoName)
            geomInfo.setGeomPropValue('udimset', sorted(self._udimNumbers), 'stringarray')

        return self.doc

    def _addTexture(self, textureFile):
        '''Connect a texture to the shader input that best matches its name.'''
        shaderInput = self.matcher.matchTexture(textureFile.name)
        if not shaderInput:
            print(f'Skipping {textureFile.name} which does not match any {self.shadingModel} input')
            return
        if not self._claimInput(shaderInput, textureFile, textureFile.name):
            return

        sourceNode = self._addImageNode(textureFile, shaderInput.getType())
        if inputRequiresNormalMap(shaderInput):
            normalMapNode = self.nodeGraph.addNode('normalmap', f'{sourceNode.getName()}_normalmap', 'vector3')
            normalMapNode.setConnectedNode('in', sourceNode)
            sourceNode = normalMapNode
        self._connect(sourceNode, shaderInput)

    def _addPackedTexture(self, textureFile, channels):
        '''Connect each channel of a packed texture to the shader input matching its term.'''
        connections = []
        for channel, term in channels:
            subject = f'{CHANNEL_LABELS[channel]} channel of {textureFile.name}'
            shaderInput = self.matcher.matchTerm(normalizeName(term))
            if not shaderInput or shaderInput.getType() != 'float':
                print(f'Skipping {subject} which does not match any {self.shadingModel} input')
            elif self._claimInput(shaderInput, textureFile, subject):
                connections.append((channel, shaderInput))
        if not connections:
            return

        # The image is read as color3 to preserve all channels, but carries no color space,
        # as packed textures store scalar data rather than colors
        imageNode = self._addImageNode(textureFile, 'color3', isColorData=False)
        for channel, shaderInput in connections:
            extractNode = self.nodeGraph.addNode('extract', f'{imageNode.getName()}_{CHANNEL_LABELS[channel]}', 'float')
            extractNode.setConnectedNode('in', imageNode)
            extractNode.setInputValue('index', channel, 'integer')
            self._connect(extractNode, shaderInput)

    def _claimInput(self, shaderInput, textureFile, subject):
        '''Record that a texture connects to a shader input, unless the input is already connected.'''
        inputName = shaderInput.getName()
        if inputName in self._connectedInputs:
            print(f'Skipping {subject} as {inputName} is already connected to {self._connectedInputs[inputName]}')
            return False
        self._connectedInputs[inputName] = textureFile.name
        return True

    def _addImageNode(self, textureFile, outputType, isColorData=True):
        '''Create an image node for a texture, with a color space for color-typed images.'''
        nodeName = self.nodeGraph.createValidChildName(textureFile.name)
        imageNode = self.nodeGraph.addNode(self.imageCategory, nodeName, outputType)
        if isColorData and outputType in ('color3', 'color4'):
            imageNode.setColorSpace(self.colorspace)
        imageNode.setInputValue('file', os.path.relpath(textureFile.pattern, self.outputDir), 'filename')
        self._udimNumbers.update(textureFile.udimNumbers)
        return imageNode

    def _connect(self, sourceNode, shaderInput):
        '''Connect a node in the graph to a shader input through a graph output.'''
        outputNode = self.nodeGraph.addOutput(f'{sourceNode.getName()}_output', shaderInput.getType())
        outputNode.setConnectedNode(sourceNode)
        inputNode = self.shaderNode.addInput(shaderInput.getName())
        inputNode.setConnectedOutput(outputNode)
        inputNode.setType(shaderInput.getType())


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def loadStandardLibraries(customPaths=None, customLibraries=None):
    '''Load and return the standard MaterialX libraries, along with any custom libraries.'''
    searchPath = mx.getDefaultDataSearchPath()
    for path in itertools.chain.from_iterable(customPaths or []):
        searchPath.append(path)

    # Custom libraries are loaded before the defaults so that they take precedence
    libraryFolders = list(itertools.chain.from_iterable(customLibraries or []))
    libraryFolders.extend(mx.getDefaultDataLibraryFolders())

    stdlib = mx.createDocument()
    mx.loadLibraries(libraryFolders, searchPath, stdlib)
    return stdlib

def findShadingModelNodeDef(stdlib, shadingModel):
    '''Find the nodedef for the requested shading model, preferring its default version.'''
    matchingNodeDefs = stdlib.getMatchingNodeDefs(shadingModel)
    for nodeDef in matchingNodeDefs:
        if nodeDef.getAttribute('isdefaultversion') == 'true':
            return nodeDef
    return matchingNodeDefs[0] if matchingNodeDefs else None

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
        '--colorSpace', dest='colorSpace', type=str, default='srgb_rec709_scene',
        help='The colorspace in which input textures should be interpreted, defaulting to srgb_rec709_scene.'
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
    stdlib = loadStandardLibraries(opts.paths, opts.libraries)
    shadingModelNodeDef = findShadingModelNodeDef(stdlib, opts.shadingModel)
    if not shadingModelNodeDef:
        print(f'Shading model {opts.shadingModel} not found in the MaterialX data libraries')
        return

    print(f'Analyzing textures in the {texturePath.asString()} folder for the {opts.shadingModel} shading model.')

    # Create the MaterialX document
    builder = MaterialBuilder(
        shadingModelNodeDef,
        materialName=mx.createValidName(mtlxFile.getBaseName().rsplit('.', 1)[0]),
        outputDir=mtlxFile.getParentPath().asString(),
        colorspace=opts.colorSpace,
        useTiledImage=opts.tiledImage,
    )
    doc = builder.build(textureFiles)

    # Output the document
    if opts.outputFilename:
        if not mtlxFile.getParentPath().exists():
            mtlxFile.getParentPath().createDirectory(True)
        mx.writeToXmlFile(doc, mtlxFile.asString())
        print(f'Wrote MaterialX document to disk: {mtlxFile.asString()}')
    else:
        print('Generated MaterialX document:')
        print(mx.writeToXmlString(doc))


if __name__ == '__main__':
    main()
