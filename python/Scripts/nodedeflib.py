'''
Shared utilities for MaterialX node definition processing.

This module provides common functionality for parsing specification documents,
handling the MaterialX type system, and working with node signatures.
'''

import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import NamedTuple
import MaterialX as mx


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class PortInfo:
    '''Information about an input or output port from the specification.'''
    name: str
    types: list = field(default_factory=list)  # Ordered list of types, preserving spec order
    typeRef: str = None  # For "Same as X" references
    default: str = None  # Spec default string (before type-specific expansion)
    description: str = None  # Port description from spec
    acceptedValues: str = None  # Accepted values from spec

@dataclass(frozen=True)
class NodeSignature:
    '''A typed combination of inputs and outputs, corresponding to one nodedef.'''
    inputs: tuple   # ((name, type), ...) sorted for hashing
    outputs: tuple  # ((name, type), ...) sorted for hashing
    _displayInputs: tuple = None
    _displayOutputs: tuple = None

    @classmethod
    def create(cls, inputs, outputs):
        '''Create a NodeSignature from input/output dicts of name -> type.'''
        return cls(
            inputs=tuple(sorted(inputs.items())),
            outputs=tuple(sorted(outputs.items())),
            _displayInputs=tuple(inputs.items()),
            _displayOutputs=tuple(outputs.items()),
        )

    def __hash__(self):
        return hash((self.inputs, self.outputs))

    def __eq__(self, other):
        if not isinstance(other, NodeSignature):
            return False
        return self.inputs == other.inputs and self.outputs == other.outputs

    def __str__(self):
        insStr = ', '.join(f'{n}:{t}' for n, t in self._displayInputs)
        outsStr = ', '.join(f'{n}:{t}' for n, t in self._displayOutputs)
        return f'({insStr}) -> {outsStr}'

@dataclass
class NodeInfo:
    '''A node and its supported signatures.'''
    name: str
    signatures: set = field(default_factory=set)
    specInputs: dict = field(default_factory=dict)  # Port info for default value comparison

class IssueType(Enum):
    '''Categories of issues found during specification parsing and comparison.'''
    # Specification validation issues
    SPEC_COLUMN_MISMATCH = 'Column Count Mismatches in Specification'
    SPEC_EMPTY_PORT_NAME = 'Empty Port Names in Specification'
    SPEC_UNRECOGNIZED_TYPE = 'Unrecognized Types in Specification'
    # Node-level differences
    NODE_MISSING_IN_LIBRARY = 'Nodes in Specification but not Data Library'
    NODE_MISSING_IN_SPEC = 'Nodes in Data Library but not Specification'
    # Signature-level differences
    SIGNATURE_DIFFERENT_INPUTS = 'Nodes with Different Input Sets'
    SIGNATURE_MISSING_IN_LIBRARY = 'Node Signatures in Specification but not Data Library'
    SIGNATURE_MISSING_IN_SPEC = 'Node Signatures in Data Library but not Specification'
    # Default value differences
    DEFAULT_MISMATCH = 'Default Value Mismatches'

@dataclass
class Issue:
    '''An issue found during specification parsing or comparison.'''
    issueType: IssueType
    nodeName: str
    portName: str = None
    details: str = None
    signature: 'NodeSignature' = None
    extraInLib: tuple = None
    extraInSpec: tuple = None
    valueType: str = None
    specDefault: str = None
    libDefault: str = None

def formatIssue(issue):
    '''Format an Issue for display, returning a list of lines.'''
    # Default mismatch
    if issue.issueType == IssueType.DEFAULT_MISMATCH:
        return [
            f'  {issue.nodeName}.{issue.portName} ({issue.valueType}):',
            f'    Signature:            {issue.signature}',
            f'    Spec default:         {issue.specDefault}',
            f'    Data library default: {issue.libDefault}',
        ]

    # Different input sets
    if issue.issueType == IssueType.SIGNATURE_DIFFERENT_INPUTS:
        lines = [f'  {issue.nodeName}: {issue.signature}']
        if issue.extraInLib:
            extraStr = ', '.join(f'{n}:{t}' for n, t in issue.extraInLib)
            lines.append(f'    Extra in library: {extraStr}')
        if issue.extraInSpec:
            extraStr = ', '.join(f'{n}:{t}' for n, t in issue.extraInSpec)
            lines.append(f'    Extra in spec:    {extraStr}')
        return lines

    # Signature mismatch (missing in spec or library)
    if issue.signature:
        return [f'  {issue.nodeName}: {issue.signature}']

    # Spec validation error with port
    if issue.portName:
        return [f'  {issue.nodeName}.{issue.portName}']

    # Node-level difference or simple spec validation error
    return [f'  {issue.nodeName}']

@dataclass
class ParsedTable:
    '''A parsed table from the specification with its port information.'''
    headers: list
    inputs: dict = field(default_factory=dict)   # name -> PortInfo
    outputs: dict = field(default_factory=dict)  # name -> PortInfo
    portOrder: list = field(default_factory=list)  # Preserve row order
    startLine: int = 0
    endLine: int = 0

@dataclass
class NodeSection:
    '''A node section from the specification document.'''
    name: str
    headerLine: int
    tables: list = field(default_factory=list)  # List of ParsedTable

# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------

class ParsedSpecResult(NamedTuple):
    '''Result of parsing a specification document.'''
    nodes: dict  # name -> NodeInfo (with expanded signatures)
    nodeSections: dict  # name -> NodeSection (with raw tables and line info)
    issues: list  # List of Issue (validation issues found during parsing)
    lines: list  # Original document lines

class DataLibraryResult(NamedTuple):
    '''Result of loading a data library MTLX document.'''
    nodes: dict
    defaults: dict
    nodedefNames: dict

class ParsedTypes(NamedTuple):
    '''Result of parsing a specification type string.'''
    types: list
    typeRef: str

class TypedDefault(NamedTuple):
    '''Result of converting a spec default to a typed value.'''
    value: object
    isGeomprop: bool

class MarkdownTableResult(NamedTuple):
    '''Result of parsing a markdown table.'''
    rows: list
    headers: list
    columnMismatchCount: int
    endLine: int

class TypeSystemContext(NamedTuple):
    '''Type system data structures for processing specifications.'''
    typeGroups: dict
    typeGroupVariables: dict
    typeGroupOrder: dict
    geompropNames: set
    specDefaultNotation: dict
    knownTypes: set


# -----------------------------------------------------------------------------
# Type System
# -----------------------------------------------------------------------------

def loadStandardLibraries():
    '''Load and return the standard MaterialX libraries as a document.'''
    stdlib = mx.createDocument()
    mx.loadLibraries(mx.getDefaultDataLibraryFolders(), mx.getDefaultDataSearchPath(), stdlib)
    return stdlib

def loadDataLibrary(mtlxPath):
    '''Load a data library MTLX document. Returns a DataLibraryResult.'''
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, str(mtlxPath))

    nodes = {}
    defaults = {}
    nodedefNames = {}

    for nodedef in doc.getNodeDefs():
        nodeName = nodedef.getNodeString()
        nodedefName = nodedef.getName()

        # Build signature
        sigInputs = {inp.getName(): inp.getType() for inp in nodedef.getInputs()}
        sigOutputs = {out.getName(): out.getType() for out in nodedef.getOutputs()}
        sig = NodeSignature.create(sigInputs, sigOutputs)

        # Add to nodes
        node = nodes.setdefault(nodeName, NodeInfo(name=nodeName))
        node.signatures.add(sig)

        # Store nodedef name
        nodedefNames[(nodeName, sig)] = nodedefName

        # Extract defaults
        sigDefaults = {}
        for inp in nodedef.getInputs():
            if inp.hasDefaultGeomPropString():
                sigDefaults[inp.getName()] = (inp.getDefaultGeomPropString(), True)
            elif inp.getValue() is not None:
                sigDefaults[inp.getName()] = (inp.getValue(), False)
        if sigDefaults:
            defaults[(nodeName, sig)] = sigDefaults

    return DataLibraryResult(nodes, defaults, nodedefNames)

def buildStandardTypes(stdlib):
    '''Build the set of standard type names from library TypeDefs.'''
    return {td.getName() for td in stdlib.getTypeDefs()}

def buildTypeGroups(stdlib):
    '''Build type groups (colorN, vectorN, matrixNN) from standard library TypeDefs.'''
    groups = {}
    for td in stdlib.getTypeDefs():
        name = td.getName()
        # Match colorN, vectorN patterns (color3, vector2, etc.)
        match = re.match(r'^(color|vector)(\d)$', name)
        if match:
            groupName = f'{match.group(1)}N'
            groups.setdefault(groupName, set()).add(name)
            continue
        # Match matrixNN pattern (matrix33, matrix44)
        match = re.match(r'^matrix(\d)\1$', name)
        if match:
            groups.setdefault('matrixNN', set()).add(name)
    return groups

def buildTypeGroupVariables(typeGroups):
    '''Build type group variables (e.g., colorM from colorN) for "must differ" constraints.'''
    variables = {}
    for groupName in typeGroups:
        if groupName.endswith('N') and not groupName.endswith('NN'):
            variantName = groupName[:-1] + 'M'
            variables[variantName] = groupName
    return variables

def buildTypeGroupOrder(typeGroups):
    '''Build ordered list of concrete types for each type group.'''
    order = {}
    for groupName, types in typeGroups.items():
        # Sort by numeric suffix for consistent ordering
        order[groupName] = sorted(types, key=lambda t: (len(t), t))
    return order

def buildGeompropNames(stdlib):
    '''Extract geomprop names from standard library GeomPropDefs.'''
    return {gpd.getName() for gpd in stdlib.getGeomPropDefs()}

def parseSpecTypes(typeStr):
    '''Parse a specification type string. Returns a ParsedTypes.'''
    if typeStr is None or not typeStr.strip():
        return ParsedTypes([], None)

    typeStr = typeStr.strip()

    # Handle "Same as X" and "Same as X or Y" references
    sameAsMatch = re.match(r'^Same as\s+`?(\w+)`?(?:\s+or\s+(.+))?$', typeStr, re.IGNORECASE)
    if sameAsMatch:
        refPort = sameAsMatch.group(1)
        extraTypes = sameAsMatch.group(2)
        extraList = []
        if extraTypes:
            extraList = parseSpecTypes(extraTypes).types
        return ParsedTypes(extraList, refPort)

    # Normalize "or" to comma: "X or Y" -> "X, Y", "X, Y, or Z" -> "X, Y, Z"
    normalized = re.sub(r',?\s+or\s+', ', ', typeStr)

    result = []
    for t in normalized.split(','):
        t = t.strip()
        if t and t not in result:  # Preserve order, avoid duplicates
            result.append(t)

    return ParsedTypes(result, None)

def expandTypeSet(types, typeCtx):
    '''Expand type groups to concrete types. Returns list of (concreteType, groupName) tuples.'''
    result = []
    for t in types:
        if t in typeCtx.typeGroups:
            if typeCtx.typeGroupOrder and t in typeCtx.typeGroupOrder:
                concreteTypes = typeCtx.typeGroupOrder[t]
            else:
                concreteTypes = typeCtx.typeGroups[t]
            for concrete in concreteTypes:
                result.append((concrete, t))
        elif t in typeCtx.typeGroupVariables:
            baseGroup = typeCtx.typeGroupVariables[t]
            if typeCtx.typeGroupOrder and baseGroup in typeCtx.typeGroupOrder:
                concreteTypes = typeCtx.typeGroupOrder[baseGroup]
            else:
                concreteTypes = typeCtx.typeGroups[baseGroup]
            for concrete in concreteTypes:
                result.append((concrete, t))
        else:
            result.append((t, None))
    return result


# -----------------------------------------------------------------------------
# Default Value Utilities
# -----------------------------------------------------------------------------

def isClosureType(typeName):
    '''Check if a type is a closure type (distribution function, shader, or material).'''
    return typeName in {
        'BSDF', 'EDF', 'VDF',
        'surfaceshader', 'volumeshader', 'lightshader', 'displacementshader',
        'material',
    }

def getComponentCount(typeName):
    '''Get the number of components for a MaterialX type, or None if unknown.'''
    if typeName in ('float', 'integer', 'boolean'):
        return 1
    match = re.match(r'^(color|vector)(\d)$', typeName)
    if match:
        return int(match.group(2))
    match = re.match(r'^matrix(\d)(\d)$', typeName)
    if match:
        return int(match.group(1)) * int(match.group(2))
    return None

def expandPlaceholder(placeholder, typeName):
    '''Expand a placeholder (0, 1, 0.5) to a type-appropriate value string.'''
    if isClosureType(typeName):
        return '' if placeholder == '0' else None

    count = getComponentCount(typeName)
    if count is None:
        return None

    if placeholder == '0':
        return 'false' if typeName == 'boolean' else ', '.join(['0'] * count)

    if placeholder == '1':
        if typeName == 'boolean':
            return 'true'
        if typeName == 'matrix33':
            return '1, 0, 0, 0, 1, 0, 0, 0, 1'
        if typeName == 'matrix44':
            return '1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1'
        return ', '.join(['1'] * count)

    if placeholder == '0.5':
        if typeName in ('integer', 'boolean'):
            return None
        return ', '.join(['0.5'] * count)

    return None

def buildSpecDefaultNotation(geompropNames):
    '''Build the mapping from spec notation (__zero__, etc.) to normalized values.'''
    notation = {
        '__zero__': '0',
        '__one__': '1',
        '__half__': '0.5',
        '__empty__': '',
    }
    for name in geompropNames:
        notation[f'_{name}_'] = name
    return notation

def buildTypeSystemContext(stdlib=None):
    '''Build all type system data structures. Returns a TypeSystemContext.'''
    if stdlib is None:
        stdlib = loadStandardLibraries()
    standardTypes = buildStandardTypes(stdlib)
    typeGroups = buildTypeGroups(stdlib)
    typeGroupVariables = buildTypeGroupVariables(typeGroups)
    typeGroupOrder = buildTypeGroupOrder(typeGroups)
    geompropNames = buildGeompropNames(stdlib)
    specDefaultNotation = buildSpecDefaultNotation(geompropNames)
    knownTypes = standardTypes | set(typeGroups.keys()) | set(typeGroupVariables.keys())
    return TypeSystemContext(
        typeGroups=typeGroups,
        typeGroupVariables=typeGroupVariables,
        typeGroupOrder=typeGroupOrder,
        geompropNames=geompropNames,
        specDefaultNotation=specDefaultNotation,
        knownTypes=knownTypes,
    )

def parseSpecDefault(value, specDefaultNotation):
    '''Parse spec default notation to normalized form. Returns None if no default specified.'''
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return specDefaultNotation.get(value, value)

def specDefaultToTypedValue(specDefault, valueType, geompropNames):
    '''Convert a normalized spec default to a typed MaterialX value. Returns a TypedDefault.'''
    if specDefault is None or specDefault == '':
        return TypedDefault(None, False)

    if specDefault in geompropNames:
        return TypedDefault(specDefault, True)

    valueStr = expandPlaceholder(specDefault, valueType)
    if valueStr is None:
        valueStr = specDefault

    try:
        return TypedDefault(mx.createValueFromStrings(valueStr, valueType), False)
    except Exception:
        return TypedDefault(None, False)

def specDefaultToString(specDefault, valueType, geompropNames):
    '''Expand a normalized spec default to a concrete string for output.'''
    if specDefault is None:
        return None

    if specDefault == '':
        return '__empty__'

    if specDefault in geompropNames:
        return specDefault

    # Preserve __zero__ notation for closure types
    if isClosureType(valueType) and specDefault == '0':
        return '__zero__'

    expansion = expandPlaceholder(specDefault, valueType)
    return expansion if expansion is not None else specDefault

def typedValueToSpecNotation(value, valueType, geompropNames):
    '''Format a typed MaterialX value using spec notation (__zero__, etc.) for display.'''
    if value is None:
        return 'None'

    if isinstance(value, str):
        if value in geompropNames:
            return f'_{value}_'
        if value == '':
            return '__zero__' if isClosureType(valueType) else '__empty__'
        return value

    # Check if typed value matches a standard placeholder
    for placeholder, notation in [('0', '__zero__'), ('1', '__one__'), ('0.5', '__half__')]:
        expansion = expandPlaceholder(placeholder, valueType)
        if expansion is None:
            continue
        try:
            if value == mx.createValueFromStrings(expansion, valueType):
                return notation
        except Exception:
            pass

    return str(value)


# -----------------------------------------------------------------------------
# Markdown Table Parsing
# -----------------------------------------------------------------------------

def stripBackticks(s):
    '''Strip wrapping backticks from a string.'''
    s = s.strip()
    if s.startswith('`') and s.endswith('`'):
        return s[1:-1]
    return s

def parseMarkdownTable(lines, startLine):
    '''Parse a markdown table. Returns a MarkdownTableResult.'''
    table = []
    headers = []
    columnMismatchCount = 0
    idx = startLine

    # Parse header row
    if idx < len(lines) and '|' in lines[idx]:
        headerLine = lines[idx].strip()
        headers = [stripBackticks(h) for h in headerLine.split('|')[1:-1]]
        idx += 1
    else:
        return MarkdownTableResult([], [], 0, startLine)

    # Skip separator row
    if idx < len(lines) and '|' in lines[idx] and '-' in lines[idx]:
        idx += 1
    else:
        return MarkdownTableResult([], [], 0, startLine)

    # Parse data rows
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or not line.startswith('|'):
            break

        cells = [stripBackticks(c) for c in line.split('|')[1:-1]]
        if len(cells) == len(headers):
            row = {headers[i].lower(): cells[i] for i in range(len(headers))}
            table.append(row)
        else:
            columnMismatchCount += 1
        idx += 1

    return MarkdownTableResult(table, headers, columnMismatchCount, idx)


# -----------------------------------------------------------------------------
# Signature Expansion
# -----------------------------------------------------------------------------

def isValidTypeGroupAssignment(driverNames, combo, typeGroupVariables):
    '''Check if type assignments satisfy group constraints. Returns a dict or None.'''
    typeAssignment = {}
    groupAssignments = {}  # groupName -> concreteType assigned to that group

    for name, (concreteType, groupName) in zip(driverNames, combo):
        typeAssignment[name] = concreteType

        # Skip constraint checking for None types (these will be resolved via typeRef)
        if concreteType is None:
            continue

        if not groupName:
            continue

        # For group variables (colorM), get the base group (colorN)
        baseGroup = typeGroupVariables.get(groupName, groupName)
        isVariable = groupName in typeGroupVariables

        # Check consistency: all uses of the same group must have same concrete type
        if groupName in groupAssignments:
            if groupAssignments[groupName] != concreteType:
                return None
        else:
            groupAssignments[groupName] = concreteType

        # For variables: must differ from base group if base is already assigned
        if isVariable and baseGroup in groupAssignments:
            if groupAssignments[baseGroup] == concreteType:
                return None

    return typeAssignment

def resolveTypeAssignment(baseAssignment, allPorts):
    '''Resolve "Same as X" references to complete port type assignments.'''
    assignment = baseAssignment.copy()

    # Iteratively resolve references (limit iterations to handle circular refs)
    for _ in range(10):
        changed = False
        for name, port in allPorts.items():
            if name in assignment:
                continue
            if port.typeRef and port.typeRef in assignment:
                assignment[name] = assignment[port.typeRef]
                changed = True
        if not changed:
            break

    # Check all ports resolved
    if set(assignment.keys()) != set(allPorts.keys()):
        return None

    return assignment

def resolvePortTypeRefs(ports):
    '''Resolve type references between ports by copying types. Modifies ports in place.'''
    # Limit iterations to handle circular refs
    for _ in range(10):
        changed = False
        for port in ports.values():
            if port.typeRef:
                refPort = ports.get(port.typeRef)
                if refPort and refPort.types:
                    for t in refPort.types:
                        if t not in port.types:
                            port.types.append(t)
                    port.typeRef = None
                    changed = True
        if not changed:
            break

def expandSpecSignatures(inputs, outputs, typeCtx):
    '''Expand spec port definitions into concrete NodeSignatures.'''
    allPorts = {**inputs, **outputs}

    # Identify driver ports and their type options
    # - Ports with explicit types (no typeRef): use those types
    # - Ports with both types AND typeRef ("Same as X or Y"): explicit types OR inherit from typeRef
    drivers = {}
    for name, port in allPorts.items():
        if port.types and not port.typeRef:
            # Normal driver: explicit types only
            drivers[name] = expandTypeSet(port.types, typeCtx)
        elif port.types and port.typeRef:
            # "Same as X or Y" pattern: explicit types OR inherit from typeRef
            expanded = expandTypeSet(port.types, typeCtx)
            expanded.append((None, None))  # None means "inherit from typeRef"
            drivers[name] = expanded

    if not drivers:
        return []

    # Generate all combinations of driver types
    driverNames = sorted(drivers.keys())
    driverTypeLists = [drivers[n] for n in driverNames]

    signatures = []
    seen = set()
    for combo in product(*driverTypeLists):
        # Validate type group constraints (skip None values which will be resolved via typeRef)
        typeAssignment = isValidTypeGroupAssignment(driverNames, combo, typeCtx.typeGroupVariables)
        if typeAssignment is None:
            continue

        # Remove None assignments - these ports will be resolved via typeRef
        typeAssignment = {k: v for k, v in typeAssignment.items() if v is not None}

        # Resolve typeRefs for this combination
        resolved = resolveTypeAssignment(typeAssignment, allPorts)
        if resolved is None:
            continue

        # Build signature
        sigInputs = {name: resolved[name] for name in inputs if name in resolved}
        sigOutputs = {name: resolved[name] for name in outputs if name in resolved}
        sig = NodeSignature.create(sigInputs, sigOutputs)

        # Only add if not already seen (preserve first occurrence order)
        if sig not in seen:
            seen.add(sig)
            signatures.append(sig)

    return signatures


# -----------------------------------------------------------------------------
# Specification Document Parsing
# -----------------------------------------------------------------------------

class SpecDocumentParser:
    '''Parser for MaterialX specification markdown documents.'''

    def __init__(self, typeCtx):
        '''Initialize the parser with type system configuration.'''
        self.typeCtx = typeCtx

        # Results
        self.nodes = {}
        self.nodeSections = {}
        self.issues = []
        self.lines = []

        # Parsing state
        self._currentNode = None
        self._currentSection = None
        self._currentTableInputs = {}
        self._currentTableOutputs = {}
        self._idx = 0

    def parse(self, specPath):
        '''Parse a specification markdown document. Returns a ParsedSpecResult.'''
        with open(specPath, 'r', encoding='utf-8') as f:
            content = f.read()

        self.lines = content.split('\n')
        self._idx = 0

        while self._idx < len(self.lines):
            line = self.lines[self._idx]

            # Try handlers in order; each returns True if it consumed the line
            if self._handleNodeHeader(line):
                continue
            if self._handleTable(line):
                continue

            self._idx += 1

        # Finalize any remaining table
        self._finalizeCurrentTable()

        return ParsedSpecResult(
            nodes=self.nodes,
            nodeSections=self.nodeSections,
            issues=self.issues,
            lines=self.lines,
        )

    def _finalizeCurrentTable(self, tableStartLine=None, tableEndLine=None, headers=None):
        '''Expand current table to signatures and add to node.'''
        if not self._currentNode:
            return
        if not self._currentTableInputs and not self._currentTableOutputs:
            return

        node = self.nodes[self._currentNode]

        # Create ParsedTable for expansion use
        if self._currentSection is not None and headers is not None:
            table = ParsedTable(
                headers=headers,
                inputs={k: PortInfo(
                    name=v.name,
                    types=v.types.copy(),
                    typeRef=v.typeRef,
                    default=v.default,
                    description=v.description,
                    acceptedValues=v.acceptedValues,
                ) for k, v in self._currentTableInputs.items()},
                outputs={k: PortInfo(
                    name=v.name,
                    types=v.types.copy(),
                    typeRef=v.typeRef,
                    default=v.default,
                    description=v.description,
                    acceptedValues=v.acceptedValues,
                ) for k, v in self._currentTableOutputs.items()},
                portOrder=list(self._currentTableInputs.keys()) + list(self._currentTableOutputs.keys()),
                startLine=tableStartLine or 0,
                endLine=tableEndLine or 0,
            )
            self._currentSection.tables.append(table)

        # Expand to signatures
        tableSigs = expandSpecSignatures(
            self._currentTableInputs, self._currentTableOutputs, self.typeCtx
        )
        node.signatures.update(tableSigs)

        # Merge input port info for default comparison (resolve types for defaults)
        allPorts = {**self._currentTableInputs, **self._currentTableOutputs}
        resolvePortTypeRefs(allPorts)
        for name, port in self._currentTableInputs.items():
            if name not in node.specInputs:
                node.specInputs[name] = port
            else:
                for t in port.types:
                    if t not in node.specInputs[name].types:
                        node.specInputs[name].types.append(t)

        # Reset table state for next table
        self._currentTableInputs = {}
        self._currentTableOutputs = {}

    def _handleNodeHeader(self, line):
        '''Check for and handle a node header line (### `nodename`).'''
        nodeMatch = re.match(r'^###\s+`([^`]+)`', line)
        if not nodeMatch:
            return False

        # Finalize previous table before switching nodes
        self._finalizeCurrentTable()

        self._currentNode = nodeMatch.group(1)
        if self._currentNode not in self.nodes:
            self.nodes[self._currentNode] = NodeInfo(name=self._currentNode)
        if self._currentNode not in self.nodeSections:
            self._currentSection = NodeSection(name=self._currentNode, headerLine=self._idx)
            self.nodeSections[self._currentNode] = self._currentSection
        else:
            self._currentSection = self.nodeSections[self._currentNode]

        self._idx += 1
        return True

    def _handleTableRow(self, row):
        '''Process a single table row, extracting port information.'''
        portName = row.get('port', '').strip('`*')

        # Track empty port names
        if not portName:
            self.issues.append(Issue(
                issueType=IssueType.SPEC_EMPTY_PORT_NAME,
                nodeName=self._currentNode,
            ))
            return

        portType = row.get('type', '')
        portDefault = row.get('default', '')
        portDesc = row.get('description', '')
        portAccepted = row.get('accepted values', '')

        types, typeRef = parseSpecTypes(portType)

        # Track unrecognized types
        unrecognized = [t for t in types if t not in self.typeCtx.knownTypes]
        if unrecognized:
            self.issues.append(Issue(
                issueType=IssueType.SPEC_UNRECOGNIZED_TYPE,
                nodeName=self._currentNode,
                portName=portName,
                details=', '.join(sorted(unrecognized)),
            ))

        # Determine if this is an output port
        isOutput = (portName == 'out') or portDesc.lower().startswith('output')
        target = self._currentTableOutputs if isOutput else self._currentTableInputs

        # Create port info for this table
        portInfo = target.setdefault(portName, PortInfo(
            name=portName,
            default=parseSpecDefault(portDefault, self.typeCtx.specDefaultNotation),
            description=portDesc,
            acceptedValues=portAccepted,
        ))
        for t in types:
            if t not in portInfo.types:
                portInfo.types.append(t)
        if typeRef and not portInfo.typeRef:
            portInfo.typeRef = typeRef

    def _handleTable(self, line):
        '''Check for and handle a table start.'''
        if not self._currentNode:
            return False
        if '|' not in line or 'Port' not in line:
            return False

        # Finalize previous table before starting new one
        self._finalizeCurrentTable()

        tableStartLine = self._idx
        rows, headers, columnMismatchCount, self._idx = parseMarkdownTable(self.lines, self._idx)
        tableEndLine = self._idx

        # Track column count mismatches
        for _ in range(columnMismatchCount):
            self.issues.append(Issue(
                issueType=IssueType.SPEC_COLUMN_MISMATCH,
                nodeName=self._currentNode,
            ))

        if rows:
            for row in rows:
                self._handleTableRow(row)

            # Finalize this table with its metadata
            self._finalizeCurrentTable(tableStartLine, tableEndLine, headers)

        return True


def parseSpecDocument(specPath, typeCtx):
    '''Parse a specification markdown document. Returns a ParsedSpecResult.'''
    parser = SpecDocumentParser(typeCtx)
    return parser.parse(specPath)


def needsExpansion(table, typeCtx):
    '''Check if a table needs expansion (has type groups or type references).'''
    allPorts = {**table.inputs, **table.outputs}
    for port in allPorts.values():
        # Check for type groups
        for t in port.types:
            if t in typeCtx.typeGroups or t in typeCtx.typeGroupVariables:
                return True
        # Check for type references
        if port.typeRef:
            return True
    return False
