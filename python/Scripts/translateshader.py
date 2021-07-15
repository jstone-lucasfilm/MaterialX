#!/usr/bin/env python
'''
Generate a baked translated version of each material in the input document, using the ShaderTranslator class in the MaterialXShaderGen library
and the TextureBaker class in the MaterialXRenderGlsl library.
'''

import sys, os, argparse
import MaterialX as mx

from MaterialX import PyMaterialXGenShader as mx_gen_shader
from MaterialX import PyMaterialXGenGlsl as ms_gen_glsl
from MaterialX import PyMaterialXRender as mx_render
from MaterialX import PyMaterialXRenderGlsl as mx_render_glsl

def greatestPowerOfTwo(num):
    power = int(math.log(num, 2))
    return int(pow(2, power))

def main():
    parser = argparse.ArgumentParser(description="Generate a translated baked version of each material in the input document.")
    parser.add_argument("--hdr", dest="hdr", action="store_true", help="Save images to hdr format.")
    parser.add_argument("--maxSize", dest="maxSize", type=int, default=0, help="Specify an optional maximum for the width and height of baked documents.")
    parser.add_argument("--powerOfTwo", dest="powerOfTwo", action="store_true", help="Optionally clip texture dimensions to the largest powers of below the computed values.")
    parser.add_argument("--path", dest="paths", action='append', nargs='+', help="An additional absolute search path location (e.g. '/projects/MaterialX')")
    parser.add_argument("--library", dest="libraries", action='append', nargs='+', help="An additional relative path to a custom data library folder (e.g. 'libraries/custom')")
    parser.add_argument(dest="inputFilename", help="Filename of the input document.")
    parser.add_argument(dest="outputFilename", help="Filename of the output document.")
    parser.add_argument(dest="destShader", help="Destination shader for translation")
    opts = parser.parse_args()

    doc = mx.createDocument()
    try:
        mx.readFromXmlFile(doc, opts.inputFilename)
    except mx.ExceptionFileMissing as err:
        print(err)
        sys.exit(0)

    stdlib = mx.createDocument()
    filePath = os.path.dirname(os.path.abspath(__file__))
    searchPath = mx.FileSearchPath(os.path.join(filePath, '..', '..'))
    searchPath.append(os.path.dirname(opts.inputFilename))
    libraryFolders = [ "libraries" ]
    if opts.paths:
        for pathList in opts.paths:
            for path in pathList:
                searchPath.append(path)
    if opts.libraries:
        for libraryList in opts.libraries:
            for library in libraryList:
                libraryFolders.append(library)
    mx.loadLibraries(libraryFolders, searchPath, stdlib)
    doc.importLibrary(stdlib)

    valid, msg = doc.validate()
    if not valid:
        print("Validation warnings for input document:")
        print(msg)

    # Translate between shading models
    translator = mx_gen_shader.ShaderTranslator.create()
    try:
        translator.translateAllMaterials(doc, opts.destShader)
    except mx.Exception as err:
        print(err)
        sys.exit(0)
        
    # Query the UDIM set of the document.
    udimSetValue = doc.getGeomPropValue('udimset')
    udimSet = udimSetValue.getData() if udimSetValue else []

    # Compute baking resolution and cache source images.
    imageHandler = mx_render.ImageHandler.create(mx_render.StbImageLoader.create())
    imageHandler.setSearchPath(searchPath)
    if udimSet:
        print('Found UDIM set:', udimSet)
        resolver = doc.createStringResolver()
        resolver.setUdimString(udimSet[0])
        imageHandler.setFilenameResolver(resolver)
    imageVec = imageHandler.getReferencedImages(doc)
    bakeWidth, bakeHeight = mx_render.getMaxDimensions(imageVec)
    if opts.maxSize > 0:
        bakeWidth = min(bakeWidth, opts.maxSize)
        bakeHeight = min(bakeHeight, opts.maxSize)
    if opts.powerOfTwo:
        bakeWidth = greatestPowerOfTwo(bakeWidth)
        bakeHeight = greatestPowerOfTwo(bakeHeight)
    bakeWidth = max(bakeWidth, 4)
    bakeHeight = max(bakeHeight, 4)
    print('Baking resolution:', bakeWidth, bakeHeight)

    # Bake the resulting material to flat textures.
    baseType = mx_render.BaseType.FLOAT if opts.hdr else mx_render.BaseType.UINT8
    baker = mx_render_glsl.TextureBaker.create(bakeWidth, bakeHeight, baseType)
    baker.bakeAllMaterials(doc, searchPath, opts.outputFilename)

if __name__ == '__main__':
    main()
