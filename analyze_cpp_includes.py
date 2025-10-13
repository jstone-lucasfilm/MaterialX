#!/usr/bin/env python
'''
Analyze C++ include dependencies and generate a DOT graph file.

This script recursively scans for C++ source and header files in a directory,
extracts #include statements, and generates a DOT file representing the
include dependency graph.
'''

import os
import re
import argparse
import time
from pathlib import Path
from collections import defaultdict


def findCppFiles(rootDir):
    '''Find all C++ source and header files recursively.'''
    cppExtensions = {'.h', '.hpp', '.hxx', '.H', '.hh',
                     '.c', '.cpp', '.cxx', '.cc', '.C', '.c++'}
    cppFiles = []

    for root, dirs, files in os.walk(rootDir):
        # Skip hidden directories and common build/dependency directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and
                   d not in {'build', 'cmake-build-debug', 'cmake-build-release',
                            'out', 'bin', 'obj', 'node_modules', 'vendor', 'third_party'}]

        for file in files:
            if any(file.endswith(ext) for ext in cppExtensions):
                fullPath = os.path.join(root, file)
                cppFiles.append(fullPath)

    return cppFiles


def extractIncludes(filePath):
    '''Extract all #include statements from a C++ file.'''
    includePattern = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)
    includes = []

    try:
        with open(filePath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            matches = includePattern.findall(content)
            includes = matches
    except Exception as err:
        print("Warning: Could not read %s: %s" % (filePath, err))

    return includes


def normalizePath(path, rootDir):
    '''Normalize a file path relative to the root directory.'''
    try:
        # Convert to Path object for easier manipulation
        pathObj = Path(path)
        rootObj = Path(rootDir)

        # Try to make it relative to root
        try:
            relPath = pathObj.relative_to(rootObj)
            return str(relPath).replace('\\', '/')
        except ValueError:
            # If not relative to root, just return the path as is
            return str(pathObj).replace('\\', '/')
    except Exception:
        return str(path).replace('\\', '/')


def findIncludedFile(includePath, currentFile, allFiles, rootDir):
    '''Try to resolve an include path to an actual file.'''
    currentDir = os.path.dirname(currentFile)

    # Try different resolution strategies
    candidates = []

    # 1. Relative to current file's directory
    candidates.append(os.path.normpath(os.path.join(currentDir, includePath)))

    # 2. Relative to root directory
    candidates.append(os.path.normpath(os.path.join(rootDir, includePath)))

    # 3. Search in common include directories
    for subdir in ['include', 'src', 'source', 'lib']:
        subdirPath = os.path.join(rootDir, subdir)
        if os.path.exists(subdirPath):
            candidates.append(os.path.normpath(os.path.join(subdirPath, includePath)))

    # 4. Check if the include path matches any file basename
    includeBasename = os.path.basename(includePath)

    # Try to find the file
    for candidate in candidates:
        if candidate in allFiles:
            return candidate
        # Try with common extensions if no extension provided
        if not any(candidate.endswith(ext) for ext in ['.h', '.hpp', '.cpp', '.c', '.cc', '.cxx']):
            for ext in ['.h', '.hpp', '.cpp', '.c', '.cc', '.cxx']:
                candidateWithExt = candidate + ext
                if candidateWithExt in allFiles:
                    return candidateWithExt

    # Last resort: search for matching basename
    for filePath in allFiles:
        if os.path.basename(filePath) == includeBasename:
            return filePath

    return None


def buildDependencyGraph(rootDir):
    '''Build the include dependency graph.'''
    print("Scanning for C++ files in %s..." % rootDir)
    cppFiles = findCppFiles(rootDir)
    print("Found %d C++ files" % len(cppFiles))

    # Create a set for faster lookups
    allFilesSet = set(cppFiles)

    # Build the graph
    graph = defaultdict(set)  # file -> set of files it includes
    externalIncludes = defaultdict(set)  # file -> set of external includes

    for i, filePath in enumerate(cppFiles, 1):
        if i % 100 == 0:
            print("Processing file %d/%d..." % (i, len(cppFiles)))

        includes = extractIncludes(filePath)
        normalizedSource = normalizePath(filePath, rootDir)

        for include in includes:
            # Try to resolve the include to a real file
            resolved = findIncludedFile(include, filePath, allFilesSet, rootDir)

            if resolved:
                normalizedTarget = normalizePath(resolved, rootDir)
                if normalizedTarget != normalizedSource:  # Avoid self-loops
                    graph[normalizedSource].add(normalizedTarget)
            else:
                # Track external includes (system headers, third-party libraries)
                externalIncludes[normalizedSource].add(include)

    return graph, externalIncludes


def generateDotFile(graph, outputFile, includeExternal=False, externalIncludes=None):
    '''Generate a DOT file from the dependency graph.'''
    with open(outputFile, 'w', encoding='utf-8') as f:
        f.write("digraph CPPIncludes {\n")
        f.write("    rankdir=LR;\n")  # Left to right layout
        f.write("    node [shape=box, style=filled, fillcolor=lightblue];\n")
        f.write("    edge [color=gray];\n")
        f.write("\n")

        # Collect all nodes
        allNodes = set()
        for source in graph:
            allNodes.add(source)
            allNodes.update(graph[source])

        # Write node declarations with shortened labels for readability
        f.write("    // Nodes\n")
        for node in sorted(allNodes):
            # Use just the filename for label, but keep full path as ID
            label = os.path.basename(node)
            # Color headers differently from source files
            if node.endswith(('.h', '.hpp', '.hxx', '.H', '.hh')):
                color = "lightblue"
            else:
                color = "lightyellow"
            f.write('    "%s" [label="%s", fillcolor=%s];\n' % (node, label, color))

        f.write("\n    // Edges\n")
        # Write edges
        edgeCount = 0
        for source in sorted(graph):
            for target in sorted(graph[source]):
                f.write('    "%s" -> "%s";\n' % (source, target))
                edgeCount += 1

        # Optionally include external dependencies
        if includeExternal and externalIncludes:
            f.write("\n    // External includes (commented out by default)\n")
            for source in sorted(externalIncludes):
                for extInclude in sorted(externalIncludes[source]):
                    f.write('    // "%s" -> "%s" [style=dashed, color=red];\n' % (source, extInclude))

        f.write("}\n")

    return len(allNodes), edgeCount


def analyzeGraphStatistics(graph):
    '''Analyze and return statistics about the dependency graph.'''
    stats = {}

    # Basic counts
    allNodes = set()
    for source in graph:
        allNodes.add(source)
        allNodes.update(graph[source])

    stats['totalFiles'] = len(allNodes)
    stats['filesWithIncludes'] = len(graph)

    # Count total edges
    totalEdges = sum(len(targets) for targets in graph.values())
    stats['totalEdges'] = totalEdges

    # Find files with most dependencies
    depsCount = [(source, len(targets)) for source, targets in graph.items()]
    depsCount.sort(key=lambda x: x[1], reverse=True)
    stats['mostDependencies'] = depsCount[:5] if depsCount else []

    # Find most included files
    includeCount = defaultdict(int)
    for targets in graph.values():
        for target in targets:
            includeCount[target] += 1

    mostIncluded = sorted(includeCount.items(), key=lambda x: x[1], reverse=True)
    stats['mostIncluded'] = mostIncluded[:5] if mostIncluded else []

    # Find isolated files (no dependencies and not included by anyone)
    includedFiles = set()
    for targets in graph.values():
        includedFiles.update(targets)

    isolated = allNodes - set(graph.keys()) - includedFiles
    stats['isolatedFiles'] = len(isolated)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze C++ include dependencies and generate a DOT graph.")
    parser.add_argument("--output", dest="output", default="cpp_includes.dot",
                       help="Output DOT file name (default: cpp_includes.dot)")
    parser.add_argument("--external", dest="external", action="store_true",
                       help="Include external dependencies as comments in the DOT file.")
    parser.add_argument("--stats", dest="stats", action="store_true",
                       help="Print detailed statistics about the dependency graph.")
    parser.add_argument(dest="directory", nargs="?", default=".",
                       help="Root directory to scan (default: current directory)")

    opts = parser.parse_args()

    rootDir = os.path.abspath(opts.directory)
    if not os.path.exists(rootDir):
        print("Error: Directory %s does not exist" % rootDir)
        return 1

    print("Analyzing C++ include dependencies in: %s" % rootDir)
    print("=" * 60)

    startTime = time.time()

    # Build the dependency graph
    graph, externalIncludes = buildDependencyGraph(rootDir)

    # Generate the DOT file
    print("\nGenerating DOT file: %s" % opts.output)
    nodeCount, edgeCount = generateDotFile(graph, opts.output, opts.external, externalIncludes)

    elapsedTime = time.time() - startTime

    print("\nDOT file generated successfully!")
    print("  - Nodes (files): %d" % nodeCount)
    print("  - Edges (includes): %d" % edgeCount)
    print("  - Output file: %s" % opts.output)
    print("  - Processing time: %.2f seconds" % elapsedTime)

    if opts.stats:
        print("\nDetailed Statistics:")
        print("=" * 60)
        stats = analyzeGraphStatistics(graph)

        print("Total files in graph: %d" % stats['totalFiles'])
        print("Files with includes: %d" % stats['filesWithIncludes'])
        print("Total include edges: %d" % stats['totalEdges'])
        print("Isolated files: %d" % stats['isolatedFiles'])

        if stats['mostDependencies']:
            print("\nFiles with most dependencies:")
            for file, count in stats['mostDependencies']:
                print("  - %s: %d includes" % (os.path.basename(file), count))

        if stats['mostIncluded']:
            print("\nMost included files:")
            for file, count in stats['mostIncluded']:
                print("  - %s: included %d times" % (os.path.basename(file), count))

    print("\nTo visualize the graph, you can use Graphviz:")
    print("  dot -Tsvg %s -o cpp_includes.svg" % opts.output)

    return 0


if __name__ == '__main__':
    main()
