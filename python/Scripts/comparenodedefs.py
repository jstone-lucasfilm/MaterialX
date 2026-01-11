#!/usr/bin/env python
'''
Compare node definitions between a specification Markdown document and a
data library MaterialX document.

Report any differences between the two in their supported node sets, typed
node signatures, and default values.
'''

import argparse
from enum import Enum
from pathlib import Path

import nodedeflib as nl


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

class MatchType(Enum):
    '''Types of signature matches between spec and library.'''
    EXACT = 'exact'  # Identical inputs and outputs
    DIFFERENT_INPUTS = 'different_inputs'  # Same outputs but different inputs


# -----------------------------------------------------------------------------
# Comparison Logic
# -----------------------------------------------------------------------------

def compareSignatureDefaults(nodeName, signature, specNode, libDefaults, typeCtx):
    '''Compare default values for a matching signature. Returns list of Issues.'''
    issues = []

    for portName, valueType in signature.inputs:
        specPort = specNode.specInputs.get(portName)
        if not specPort or not specPort.default:
            continue

        specValue, specIsGeomprop = nl.specDefaultToTypedValue(specPort.default, valueType, typeCtx.geompropNames)
        libValue, libIsGeomprop = libDefaults.get(portName, (None, False))

        # Skip if either value is unavailable
        if specValue is None or libValue is None:
            continue

        # Compare values (geomprops compare as strings, typed values use equality)
        valuesMatch = (specValue == libValue) if (specIsGeomprop == libIsGeomprop) else False

        if not valuesMatch:
            issues.append(nl.Issue(
                issueType=nl.IssueType.DEFAULT_MISMATCH,
                nodeName=nodeName,
                portName=portName,
                signature=signature,
                valueType=valueType,
                specDefault=nl.typedValueToSpecNotation(specValue, valueType, typeCtx.geompropNames),
                libDefault=nl.typedValueToSpecNotation(libValue, valueType, typeCtx.geompropNames),
            ))

    return issues

def findLibraryMatch(specSig, libSigs):
    '''Find a matching library signature. Returns (matchType, libSig, extraInLib, extraInSpec).'''
    specInputs = set(specSig.inputs)
    specOutputs = set(specSig.outputs)

    for libSig in libSigs:
        libInputs = set(libSig.inputs)
        libOutputs = set(libSig.outputs)

        # Check for exact match
        if specInputs == libInputs and specOutputs == libOutputs:
            return MatchType.EXACT, libSig, None, None

        # Check for different input sets (same outputs, different inputs)
        if specOutputs == libOutputs and specInputs != libInputs:
            # If there are common inputs, verify they have the same types
            commonInputNames = {name for name, _ in specInputs} & {name for name, _ in libInputs}
            if commonInputNames:
                specInputDict = dict(specSig.inputs)
                libInputDict = dict(libSig.inputs)
                typesMatch = all(specInputDict[n] == libInputDict[n] for n in commonInputNames)
                if not typesMatch:
                    continue  # Common inputs have different types - not a match

            extraInLib = tuple(sorted(libInputs - specInputs))
            extraInSpec = tuple(sorted(specInputs - libInputs))
            return MatchType.DIFFERENT_INPUTS, libSig, extraInLib, extraInSpec

    return None, None, None, None

def compareNodes(specNodes, libNodes, libDefaults, typeCtx, compareDefaults=False):
    '''Compare nodes between spec and library. Returns list of Issues.'''
    issues = []

    specNames = set(specNodes.keys())
    libNames = set(libNodes.keys())

    # Nodes in spec but not in library
    for nodeName in sorted(specNames - libNames):
        issues.append(nl.Issue(
            issueType=nl.IssueType.NODE_MISSING_IN_LIBRARY,
            nodeName=nodeName))

    # Nodes in library but not in spec
    for nodeName in sorted(libNames - specNames):
        issues.append(nl.Issue(
            issueType=nl.IssueType.NODE_MISSING_IN_SPEC,
            nodeName=nodeName))

    # Compare signatures for common nodes
    for nodeName in sorted(specNames & libNames):
        specNode = specNodes[nodeName]
        libNode = libNodes[nodeName]

        specSigs = specNode.signatures
        libSigs = libNode.signatures

        # Track which signatures have been matched
        matchedLibSigs = set()
        matchedSpecSigs = set()
        inputDiffMatches = []  # (specSig, libSig, extraInLib, extraInSpec)

        # For each spec signature, find matching library signature
        for specSig in specSigs:
            matchType, libSig, extraInLib, extraInSpec = findLibraryMatch(specSig, libSigs)

            if matchType == MatchType.EXACT:
                matchedLibSigs.add(libSig)
                matchedSpecSigs.add(specSig)
                # Compare defaults for exact matches
                if compareDefaults:
                    sigDefaults = libDefaults.get((nodeName, libSig), {})
                    issues.extend(compareSignatureDefaults(
                        nodeName, specSig, specNode, sigDefaults, typeCtx))

            elif matchType == MatchType.DIFFERENT_INPUTS:
                matchedLibSigs.add(libSig)
                matchedSpecSigs.add(specSig)
                inputDiffMatches.append((specSig, libSig, extraInLib, extraInSpec))
                # Compare defaults for different input matches too (for common ports)
                if compareDefaults:
                    sigDefaults = libDefaults.get((nodeName, libSig), {})
                    issues.extend(compareSignatureDefaults(
                        nodeName, specSig, specNode, sigDefaults, typeCtx))

        # Report different input set matches
        for specSig, libSig, extraInLib, extraInSpec in sorted(inputDiffMatches, key=lambda x: str(x[0])):
            issues.append(nl.Issue(
                issueType=nl.IssueType.SIGNATURE_DIFFERENT_INPUTS,
                nodeName=nodeName,
                signature=specSig,
                extraInLib=extraInLib,
                extraInSpec=extraInSpec,
            ))

        # Spec signatures not matched by any library signature
        for specSig in sorted(specSigs - matchedSpecSigs, key=str):
            issues.append(nl.Issue(
                issueType=nl.IssueType.SIGNATURE_MISSING_IN_LIBRARY,
                nodeName=nodeName,
                signature=specSig,
            ))

        # Library signatures not matched by any spec signature
        for libSig in sorted(libSigs - matchedLibSigs, key=str):
            issues.append(nl.Issue(
                issueType=nl.IssueType.SIGNATURE_MISSING_IN_SPEC,
                nodeName=nodeName,
                signature=libSig,
            ))

    return issues


# -----------------------------------------------------------------------------
# Output Formatting
# -----------------------------------------------------------------------------

def printIssues(issues):
    '''Print the issues in a formatted way.'''
    if not issues:
        print("No differences found between specification and data library.")
        return

    # Group issues by type
    byType = {}
    for issue in issues:
        byType.setdefault(issue.issueType, []).append(issue)

    print(f"\n{'=' * 70}")
    print(f"COMPARISON RESULTS: {len(issues)} difference(s) found")
    print(f"{'=' * 70}")

    for issueType in nl.IssueType:
        if issueType not in byType:
            continue

        typeIssues = byType[issueType]
        print(f"\n{issueType.value} ({len(typeIssues)}):")
        print("-" * 50)

        for issue in typeIssues:
            for line in nl.formatIssue(issue):
                print(line)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare node definitions between a specification Markdown document and a data library MaterialX document.")
    parser.add_argument('--spec', dest='specFile',
        help='Path to the specification Markdown document. Defaults to documents/Specification/MaterialX.StandardNodes.md')
    parser.add_argument('--mtlx', dest='mtlxFile',
        help='Path to the data library MaterialX document. Defaults to libraries/stdlib/stdlib_defs.mtlx')
    parser.add_argument('--defaults', dest='compareDefaults', action='store_true',
        help='Compare default values for inputs using MaterialX typed value comparison')
    parser.add_argument('--listNodes', dest='listNodes', action='store_true',
        help='List all nodes and their node signature counts')
    opts = parser.parse_args()

    # Determine file paths
    repoRoot = Path(__file__).resolve().parent.parent.parent

    specPath = Path(opts.specFile) if opts.specFile else repoRoot / 'documents' / 'Specification' / 'MaterialX.StandardNodes.md'
    mtlxPath = Path(opts.mtlxFile) if opts.mtlxFile else repoRoot / 'libraries' / 'stdlib' / 'stdlib_defs.mtlx'

    # Verify files exist
    if not specPath.exists():
        raise FileNotFoundError(f"Specification document not found: {specPath}")

    if not mtlxPath.exists():
        raise FileNotFoundError(f"MTLX document not found: {mtlxPath}")

    print(f"Comparing:")
    print(f"  Specification: {specPath}")
    print(f"  Data Library: {mtlxPath}")

    # Build type system data
    typeCtx = nl.buildTypeSystemContext()

    # Parse specification
    print("\nParsing specification...")
    parsedSpec = nl.parseSpecDocument(specPath, typeCtx)
    specNodes = parsedSpec.nodes
    specSigCount = sum(len(n.signatures) for n in specNodes.values())
    print(f"  Found {len(specNodes)} nodes with {specSigCount} node signatures")
    if parsedSpec.issues:
        print(f"  Found {len(parsedSpec.issues)} invalid specification entries")

    # Load data library
    print("Loading data library...")
    libNodes, libDefaults, _ = nl.loadDataLibrary(mtlxPath)
    libSigCount = sum(len(n.signatures) for n in libNodes.values())
    print(f"  Found {len(libNodes)} nodes with {libSigCount} node signatures")

    # List nodes if requested
    if opts.listNodes:
        print("\nNodes in Specification:")
        for name in sorted(specNodes.keys()):
            node = specNodes[name]
            print(f"  {name}: {len(node.signatures)} signature(s)")

        print("\nNodes in Data Library:")
        for name in sorted(libNodes.keys()):
            node = libNodes[name]
            print(f"  {name}: {len(node.signatures)} signature(s)")

    # Compare nodes
    print("\nComparing node signatures...")
    issues = compareNodes(specNodes, libNodes, libDefaults, typeCtx, opts.compareDefaults)

    # Include spec validation issues
    issues = parsedSpec.issues + issues

    # Print issues
    printIssues(issues)


if __name__ == '__main__':
    main()
