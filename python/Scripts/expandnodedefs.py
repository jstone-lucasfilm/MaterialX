#!/usr/bin/env python
'''
Expand node definition tables in a specification Markdown document.

For each node with type groups (colorN, vectorN, etc.) or "Same as" type
references, generates explicit tables for each concrete type signature.
'''

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import nodedeflib as nl


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class ExpandedPort:
    '''A port with concrete type and expanded default value.'''
    name: str
    portType: str
    default: str
    description: str
    acceptedValues: str = ''


# -----------------------------------------------------------------------------
# Markdown Generation
# -----------------------------------------------------------------------------

def generateMarkdownTable(ports, headers):
    '''Generate a markdown table from a list of ExpandedPort objects.'''

    # First pass: collect all cell values for all rows
    allRows = []
    for port in ports:
        cells = []
        for h in headers:
            hLower = h.lower()
            if hLower == 'port':
                cells.append(f'`{port.name}`')
            elif hLower == 'type':
                cells.append(port.portType)
            elif hLower == 'default':
                cells.append(port.default if port.default else '')
            elif hLower == 'description':
                cells.append(port.description if port.description else '')
            elif hLower == 'accepted values':
                cells.append(port.acceptedValues if port.acceptedValues else '')
            else:
                cells.append('')
        allRows.append(cells)

    # Calculate column widths (max of header and all cell values)
    colWidths = []
    for colIdx, header in enumerate(headers):
        maxWidth = len(header)
        for row in allRows:
            if colIdx < len(row):
                maxWidth = max(maxWidth, len(row[colIdx]))
        colWidths.append(maxWidth)

    # Build header row with padding
    headerCells = [h.ljust(colWidths[i]) for i, h in enumerate(headers)]
    headerRow = '|' + '|'.join(headerCells) + '|'

    # Build separator row with matching widths
    separatorRow = '|' + '|'.join(['-' * w for w in colWidths]) + '|'

    # Build data rows with padding
    lines = [headerRow, separatorRow]
    for row in allRows:
        paddedCells = [row[i].ljust(colWidths[i]) for i in range(len(headers))]
        lines.append('|' + '|'.join(paddedCells) + '|')

    return lines


def formatSignatureTypes(sig):
    '''Format a signature as "inputTypes -> outputTypes" for subheader display.'''
    inTypes = [t for _, t in sig._displayInputs] if sig._displayInputs else []
    outTypes = [t for _, t in sig._displayOutputs] if sig._displayOutputs else []

    inStr = ', '.join(inTypes)
    outStr = ', '.join(outTypes)

    if inStr:
        return f'{inStr} -> {outStr}'
    else:
        return outStr  # Output-only nodes like `time`


# -----------------------------------------------------------------------------
# Table Expansion
# -----------------------------------------------------------------------------

def expandTableToSignatures(table, nodeName, typeCtx, nodedefNames=None):
    '''Expand a parsed table into multiple tables, one per concrete signature.'''
    # Get signatures in order (preserves spec order)
    signatures = nl.expandSpecSignatures(table.inputs, table.outputs, typeCtx)

    if not signatures:
        return [], []

    results = []
    missingSignatures = []
    allPorts = {**table.inputs, **table.outputs}

    for sig in signatures:
        # If nodedefNames is provided, check if signature exists in library
        nodedefName = None
        if nodedefNames is not None:
            nodedefName = nodedefNames.get((nodeName, sig))
            if nodedefName is None:
                missingSignatures.append(sig)

        # Build type assignment for this signature
        typeAssignment = dict(sig.inputs) | dict(sig.outputs)

        # Generate expanded ports in original order
        expandedPorts = []
        for portName in table.portOrder:
            port = allPorts.get(portName)
            if not port:
                continue

            concreteType = typeAssignment.get(portName, '')

            # Expand default value to concrete type
            expandedDefault = ''
            if port.default is not None:
                expandedDefault = nl.specDefaultToString(port.default, concreteType, typeCtx.geompropNames)
                if expandedDefault is None:
                    expandedDefault = port.default

            expandedPorts.append(ExpandedPort(
                name=portName,
                portType=concreteType,
                default=expandedDefault,
                description=port.description or '',
                acceptedValues=port.acceptedValues or '',
            ))

        results.append((sig, expandedPorts, nodedefName))

    return results, missingSignatures


# -----------------------------------------------------------------------------
# Document Transformation
# -----------------------------------------------------------------------------

def expandSpecDocument(specPath, typeCtx, nodedefNames=None):
    '''Expand a specification document. Returns (transformedContent, allMissingSignatures).'''
    parsedSpec = nl.parseSpecDocument(specPath, typeCtx)

    # Build list of (startLine, endLine, replacementLines) for all expansions
    replacements = []
    allMissingSignatures = []  # List of (nodeName, signature) tuples

    for nodeName, section in parsedSpec.nodeSections.items():
        for table in section.tables:
            # Skip tables that don't need expansion unless we're adding nodedef names
            needsExpansion = nl.needsExpansion(table, typeCtx)
            if not needsExpansion and nodedefNames is None:
                continue

            # Expand this table
            expanded, missingSignatures = expandTableToSignatures(
                table, nodeName, typeCtx, nodedefNames
            )

            # Track missing signatures
            for sig in missingSignatures:
                allMissingSignatures.append((nodeName, sig))

            if not expanded:
                continue

            # Generate replacement lines
            replacementLines = []
            for i, (sig, ports, nodedefName) in enumerate(expanded):
                if i > 0:
                    replacementLines.append('')  # Blank line between tables
                # Add subheader with nodedef name if provided
                if nodedefName is not None:
                    replacementLines.append(f'#### `{nodedefName}`')
                    replacementLines.append('')  # Blank line after subheader
                elif nodedefNames is not None:
                    # Missing nodedef - show signature types instead
                    replacementLines.append(f'#### No NodeDef: {formatSignatureTypes(sig)}')
                    replacementLines.append('')  # Blank line after subheader
                tableLines = generateMarkdownTable(ports, table.headers)
                replacementLines.extend(tableLines)

            replacements.append((table.startLine, table.endLine, replacementLines))

    # Apply replacements in reverse order to preserve line numbers
    resultLines = parsedSpec.lines.copy()
    for startLine, endLine, replacementLines in sorted(replacements, reverse=True):
        resultLines[startLine:endLine] = replacementLines

    return '\n'.join(resultLines), allMissingSignatures


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand node definition tables in a specification Markdown document.")
    parser.add_argument('--spec', dest='specFile',
        help='Path to the specification Markdown document. Defaults to documents/Specification/MaterialX.StandardNodes.md')
    parser.add_argument('--mtlx', dest='mtlxFile',
        help='Path to the data library MaterialX document. If provided, adds subheaders with matching nodedef names.')
    parser.add_argument('--output', dest='outputFile',
        help='Path to write the expanded document. Defaults to stdout.')
    parser.add_argument('--inplace', dest='inplace', action='store_true',
        help='Modify the specification file in place')
    opts = parser.parse_args()

    # Determine file paths
    repoRoot = Path(__file__).resolve().parent.parent.parent
    specPath = Path(opts.specFile) if opts.specFile else repoRoot / 'documents' / 'Specification' / 'MaterialX.StandardNodes.md'

    # Verify input file exists
    if not specPath.exists():
        raise FileNotFoundError(f"Specification document not found: {specPath}")

    # Build type system data
    typeCtx = nl.buildTypeSystemContext()

    # Optionally load nodedef names from data library
    nodedefNames = None
    if opts.mtlxFile:
        mtlxPath = Path(opts.mtlxFile)
        if not mtlxPath.exists():
            raise FileNotFoundError(f"Data library document not found: {mtlxPath}")
        print(f"Loading nodedef names from: {mtlxPath}", file=sys.stderr)
        _, _, nodedefNames = nl.loadDataLibrary(mtlxPath)
        print(f"  Found {len(nodedefNames)} nodedefs", file=sys.stderr)

    # Expand the document
    expandedContent, missingSignatures = expandSpecDocument(specPath, typeCtx, nodedefNames)

    # Report missing signatures if --mtlx was provided
    if nodedefNames is not None and missingSignatures:
        print(f"\nSignatures in specification but not in data library ({len(missingSignatures)}):", file=sys.stderr)
        for nodeName, sig in missingSignatures:
            print(f"  {nodeName}: {sig}", file=sys.stderr)

    # Output
    if opts.inplace:
        with open(specPath, 'w', encoding='utf-8') as f:
            f.write(expandedContent)
        print(f"Expanded specification written to: {specPath}", file=sys.stderr)
    elif opts.outputFile:
        outputPath = Path(opts.outputFile)
        with open(outputPath, 'w', encoding='utf-8') as f:
            f.write(expandedContent)
        print(f"Expanded specification written to: {outputPath}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding='utf-8')
        print(expandedContent)


if __name__ == '__main__':
    main()
