#!/usr/bin/env python3
"""
Alteryx Workflow to AI-Agent Analyzer - Jupyter Notebook Version
===============================================================

This notebook analyzes Alteryx workflows (.yxmd files) and converts them into:
1. Step-by-step AI-Agent implementation outlines
2. Mermaid flowchart diagrams representing the AI-Agent workflow

Author: AI Assistant
Date: 2025-09-26
"""

# ============================================================================
# CELL 1: Install Dependencies and Imports
# ============================================================================

# Uncomment the following line if running in Colab or need to install packages
# !pip install openai pathlib2 xmltodict

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import openai
from openai import OpenAI
import re
import logging
import os
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Dependencies loaded successfully!")

# ============================================================================
# CELL 2: Configuration and Constants
# ============================================================================

# Tool mappings from Alteryx plugins to readable names
TOOL_MAPPINGS = {
    'AlteryxBasePluginsGui.TextInput.TextInput': 'Text Input',
    'AlteryxSpatialPluginsGui.Summarize.Summarize': 'Summarize',
    'AlteryxBasePluginsGui.AppendFields.AppendFields': 'Append Fields',
    'AlteryxBasePluginsGui.Formula.Formula': 'Formula',
    'AlteryxBasePluginsGui.BrowseV2.BrowseV2': 'Browse',
    'AlteryxBasePluginsGui.DbFileInput.DbFileInput': 'Input Data',
    'AlteryxBasePluginsGui.DbFileOutput.DbFileOutput': 'Output Data',
    'AlteryxBasePluginsGui.Filter.Filter': 'Filter',
    'AlteryxBasePluginsGui.Join.Join': 'Join',
    'AlteryxBasePluginsGui.Union.Union': 'Union',
    'AlteryxBasePluginsGui.Sort.Sort': 'Sort',
    'AlteryxBasePluginsGui.Sample.Sample': 'Sample',
    'AlteryxBasePluginsGui.Unique.Unique': 'Unique',
    'AlteryxBasePluginsGui.CrossTab.CrossTab': 'Cross Tab',
    'AlteryxBasePluginsGui.Transpose.Transpose': 'Transpose',
    'AlteryxBasePluginsGui.AlteryxSelect.AlteryxSelect': 'Select',
    'AlteryxBasePluginsGui.Multi-FieldFormula.Multi-FieldFormula': 'Multi-Field Formula',
    'AlteryxBasePluginsGui.RecordID.RecordID': 'Record ID',
    'AlteryxBasePluginsGui.TextToColumns.TextToColumns': 'Text to Columns',
    'AlteryxConnectorGui.Download.Download': 'Download',
    'AlteryxSpatialPluginsGui.DistancePoints.DistancePoints': 'Distance',
    # Add more mappings as needed - you can expand this based on your workflows
}

# Agent mapping from tool types to AI agent names
AGENT_MAPPING = {
    "Text Input": "Data Ingestion Agent",
    "Input Data": "Data Source Agent", 
    "Summarize": "Aggregation Agent",
    "Formula": "Calculation Agent",
    "Join": "Data Join Agent",
    "Union": "Data Merge Agent",
    "Filter": "Data Filtering Agent",
    "Sort": "Data Sorting Agent",
    "Browse": "Output Validation Agent",
    "Output Data": "Data Export Agent",
    "Append Fields": "Data Enrichment Agent",
    "Unique": "Deduplication Agent",
    "Cross Tab": "Pivot Agent",
    "Transpose": "Data Transformation Agent"
}

# AI capabilities for each agent type
AGENT_CAPABILITIES = {
    "Data Ingestion Agent": [
        "Schema inference and validation",
        "Data type detection",
        "Encoding detection",
        "Format recognition",
        "Quality assessment"
    ],
    "Aggregation Agent": [
        "Statistical analysis",
        "Outlier detection",
        "Performance optimization",
        "Memory management",
        "Parallel processing"
    ],
    "Calculation Agent": [
        "Expression optimization",
        "Mathematical validation",
        "Error handling (division by zero, etc.)",
        "Precision management",
        "Formula auditing"
    ],
    "Data Join Agent": [
        "Fuzzy matching algorithms",
        "Join optimization",
        "Relationship discovery",
        "Duplicate resolution",
        "Performance tuning"
    ]
}

print("Configuration loaded successfully!")

# ============================================================================
# CELL 3: Data Structures
# ============================================================================

@dataclass
class WorkflowNode:
    """Represents a single node in an Alteryx workflow"""
    tool_id: str
    tool_type: str
    position: Tuple[int, int]
    properties: Dict
    configuration: Dict
    annotation: str = ""

@dataclass
class WorkflowConnection:
    """Represents a connection between nodes"""
    source_tool_id: str
    target_tool_id: str
    source_connection: str = "Output"
    target_connection: str = "Input"

@dataclass
class AIAgentStep:
    """Represents an AI agent processing step"""
    step_id: str
    agent_name: str
    description: str
    inputs: List[str]
    outputs: List[str]
    ai_capabilities: List[str]
    error_handling: str
    monitoring: str

print("Data structures defined successfully!")

# ============================================================================
# CELL 4: Workflow Parsing Functions
# ============================================================================

def xml_to_dict(element: ET.Element) -> Dict:
    """Convert XML element to dictionary"""
    result = {}
    for child in element:
        if len(child) == 0:
            result[child.tag] = child.text
        else:
            result[child.tag] = xml_to_dict(child)
    return result

def parse_node(node_element: ET.Element, nodes: Dict[str, WorkflowNode]):
    """Parse a single node from the XML"""
    tool_id = node_element.get('ToolID')
    
    # Get GUI settings
    gui_settings = node_element.find('GuiSettings')
    plugin = gui_settings.get('Plugin') if gui_settings is not None else 'Unknown'
    tool_type = TOOL_MAPPINGS.get(plugin, plugin)
    
    # Get position
    position_elem = gui_settings.find('Position') if gui_settings is not None else None
    position = (0, 0)
    if position_elem is not None:
        x = int(position_elem.get('x', 0))
        y = int(position_elem.get('y', 0))
        position = (x, y)
    
    # Get properties and configuration
    properties_elem = node_element.find('Properties')
    properties = {}
    configuration = {}
    annotation = ""
    
    if properties_elem is not None:
        config_elem = properties_elem.find('Configuration')
        if config_elem is not None:
            configuration = xml_to_dict(config_elem)
        
        annotation_elem = properties_elem.find('Annotation/DefaultAnnotationText')
        if annotation_elem is not None and annotation_elem.text:
            annotation = annotation_elem.text.strip()
    
    # Create workflow node
    workflow_node = WorkflowNode(
        tool_id=tool_id,
        tool_type=tool_type,
        position=position,
        properties=properties,
        configuration=configuration,
        annotation=annotation
    )
    
    nodes[tool_id] = workflow_node

def parse_connection(connection_element: ET.Element, connections: List[WorkflowConnection]):
    """Parse a single connection from the XML"""
    origin = connection_element.find('Origin')
    destination = connection_element.find('Destination')
    
    if origin is not None and destination is not None:
        source_tool_id = origin.get('ToolID')
        target_tool_id = destination.get('ToolID')
        source_connection = origin.get('Connection', 'Output')
        target_connection = destination.get('Connection', 'Input')
        
        connection = WorkflowConnection(
            source_tool_id=source_tool_id,
            target_tool_id=target_tool_id,
            source_connection=source_connection,
            target_connection=target_connection
        )
        
        connections.append(connection)

def parse_workflow_file(file_path: str) -> Tuple[Dict[str, WorkflowNode], List[WorkflowConnection]]:
    """Parse an Alteryx workflow file and extract nodes and connections"""
    nodes = {}
    connections = []
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Parse nodes
        nodes_element = root.find('Nodes')
        if nodes_element is not None:
            for node in nodes_element.findall('Node'):
                parse_node(node, nodes)
        
        # Parse connections
        connections_element = root.find('Connections')
        if connections_element is not None:
            for connection in connections_element.findall('Connection'):
                parse_connection(connection, connections)
        
        return nodes, connections
        
    except ET.ParseError as e:
        logger.error(f"Error parsing XML file: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"Workflow file not found: {e}")
        raise

print("Parsing functions defined successfully!")

# ============================================================================
# CELL 5: AI Analysis Functions
# ============================================================================

def create_workflow_summary(nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection]) -> str:
    """Create a human-readable summary of the workflow"""
    summary = []
    summary.append("Alteryx Workflow Analysis:")
    summary.append(f"Total Nodes: {len(nodes)}")
    summary.append(f"Total Connections: {len(connections)}")
    summary.append("")
    
    # List all nodes with their types and configurations
    for tool_id, node in nodes.items():
        summary.append(f"Node {tool_id}: {node.tool_type}")
        if node.annotation:
            summary.append(f"  Annotation: {node.annotation}")
        if node.configuration:
            config_str = str(node.configuration)[:200] + "..." if len(str(node.configuration)) > 200 else str(node.configuration)
            summary.append(f"  Configuration: {config_str}")
        summary.append("")
    
    # List connections
    summary.append("Workflow Flow:")
    for conn in connections:
        source_type = nodes[conn.source_tool_id].tool_type
        target_type = nodes[conn.target_tool_id].tool_type
        summary.append(f"  {source_type} (ID: {conn.source_tool_id}) -> {target_type} (ID: {conn.target_tool_id})")
    
    return "\n".join(summary)

def create_analysis_prompt(workflow_summary: str) -> str:
    """Create prompt for ChatGPT analysis"""
    return f"""
Analyze this Alteryx workflow and design an AI-agent implementation:

{workflow_summary}

Please provide:
1. A step-by-step breakdown of how this workflow would be implemented using AI agents
2. For each step, identify:
   - The AI agent type needed
   - Key capabilities required
   - Input and output data
   - Error handling strategies
   - Monitoring requirements

3. Suggest improvements that AI agents could bring over traditional ETL:
   - Intelligent error recovery
   - Adaptive processing
   - Performance optimization
   - Data quality enhancement

Format your response as a structured analysis with clear sections for each processing step.
"""

def map_node_to_agent(tool_type: str) -> str:
    """Map Alteryx tool types to AI agent names"""
    return AGENT_MAPPING.get(tool_type, f'{tool_type} Agent')

def get_agent_capabilities(tool_type: str, configuration: Dict) -> List[str]:
    """Get AI capabilities for each agent type"""
    base_capabilities = [
        "Data validation",
        "Error detection",
        "Performance monitoring",
        "Quality assurance"
    ]
    
    specific_capabilities = {
        'Text Input': ['Schema inference', 'Data type detection'],
        'Input Data': ['Format recognition', 'Encoding detection', 'Schema validation'],
        'Summarize': ['Statistical analysis', 'Outlier detection', 'Aggregation optimization'],
        'Append Fields': ['Schema alignment', 'Conflict resolution', 'Data lineage tracking'],
        'Formula': ['Expression optimization', 'Mathematical validation', 'Division by zero handling'],
        'Browse': ['Result validation', 'Data profiling', 'Quality scoring'],
        'Filter': ['Intelligent filtering', 'Pattern recognition', 'Anomaly detection'],
        'Join': ['Fuzzy matching', 'Join optimization', 'Relationship discovery']
    }
    
    return base_capabilities + specific_capabilities.get(tool_type, [])

def analyze_workflow_with_ai(nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection], api_key: str, model: str = "gpt-4") -> Dict:
    """Analyze workflow and generate AI-Agent implementation plan"""
    
    # Create workflow summary
    workflow_summary = create_workflow_summary(nodes, connections)
    
    # Generate AI-Agent analysis using ChatGPT
    prompt = create_analysis_prompt(workflow_summary)
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in data processing workflows and AI agent architecture. Analyze Alteryx workflows and design AI-agent implementations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.3
        )
        
        analysis_text = response.choices[0].message.content
        return parse_ai_analysis(analysis_text, nodes, connections)
        
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        # Fallback to rule-based analysis
        return fallback_analysis(nodes, connections)

def parse_ai_analysis(analysis_text: str, nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection]) -> Dict:
    """Parse ChatGPT analysis into structured format"""
    
    ai_agents = []
    
    # Create agents based on the workflow nodes
    for i, (tool_id, node) in enumerate(nodes.items()):
        agent_name = map_node_to_agent(node.tool_type)
        capabilities = get_agent_capabilities(node.tool_type, node.configuration)
        
        agent_step = AIAgentStep(
            step_id=f"step_{i+1}",
            agent_name=agent_name,
            description=f"Process {node.tool_type} operation",
            inputs=[],
            outputs=[],
            ai_capabilities=capabilities,
            error_handling="Intelligent retry with data validation",
            monitoring="Real-time performance and quality metrics"
        )
        
        ai_agents.append(agent_step)
    
    return {
        "ai_agents": ai_agents,
        "workflow_summary": analysis_text,
        "improvements": [
            "Intelligent error recovery",
            "Adaptive data processing",
            "Automated data quality checks",
            "Performance optimization",
            "Predictive resource management"
        ]
    }

def fallback_analysis(nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection]) -> Dict:
    """Fallback analysis when API is unavailable"""
    ai_agents = []
    
    for i, (tool_id, node) in enumerate(nodes.items()):
        agent_name = map_node_to_agent(node.tool_type)
        capabilities = get_agent_capabilities(node.tool_type, node.configuration)
        
        agent_step = AIAgentStep(
            step_id=f"step_{i+1}",
            agent_name=agent_name,
            description=f"AI-powered {node.tool_type} processing",
            inputs=[],
            outputs=[],
            ai_capabilities=capabilities,
            error_handling="Automated error detection and recovery",
            monitoring="Comprehensive performance monitoring"
        )
        
        ai_agents.append(agent_step)
    
    return {
        "ai_agents": ai_agents,
        "workflow_summary": "Automated analysis (API unavailable)",
        "improvements": [
            "Intelligent data validation",
            "Adaptive processing logic",
            "Automated error recovery",
            "Performance optimization",
            "Quality assurance"
        ]
    }

print("AI analysis functions defined successfully!")

# ============================================================================
# CELL 6: Mermaid Diagram Generation
# ============================================================================

def generate_mermaid_diagram(nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection], ai_agents: List[AIAgentStep]) -> str:
    """Generate Mermaid flowchart that accurately represents the actual workflow structure"""
    
    mermaid_code = []
    mermaid_code.append("flowchart TD")
    mermaid_code.append("    %% AI Agent Workflow Implementation")
    mermaid_code.append("")
    
    # Create a mapping of tool_id to agent for easy lookup
    agent_map = {f"step_{i+1}": agent for i, agent in enumerate(ai_agents)}
    
    # Add nodes based on actual workflow structure
    for tool_id, node in nodes.items():
        # Create a clean node identifier
        node_id = f"N{tool_id}"
        agent_name = map_node_to_agent(node.tool_type)
        
        # Add annotation if available for more context
        display_name = agent_name
        if node.annotation:
            display_name = f"{agent_name}\\n{node.annotation[:30]}{'...' if len(node.annotation) > 30 else ''}"
        
        mermaid_code.append(f"    {node_id}[\"{display_name}\"]")
    
    mermaid_code.append("")
    mermaid_code.append("    %% Workflow Connections")
    
    # Add actual connections from the workflow
    for conn in connections:
        source_id = f"N{conn.source_tool_id}"
        target_id = f"N{conn.target_tool_id}"
        
        # Add connection with labels if specific connection types
        if conn.source_connection != "Output" or conn.target_connection != "Input":
            mermaid_code.append(f"    {source_id} -->|{conn.source_connection}→{conn.target_connection}| {target_id}")
        else:
            mermaid_code.append(f"    {source_id} --> {target_id}")
    
    # Add AI capabilities as hover/subtitle information
    mermaid_code.append("")
    mermaid_code.append("    %% AI Enhancement Capabilities")
    
    # Group capabilities by agent type to avoid repetition
    unique_agents = {}
    for node in nodes.values():
        agent_type = map_node_to_agent(node.tool_type)
        if agent_type not in unique_agents:
            unique_agents[agent_type] = get_agent_capabilities(node.tool_type, node.configuration)
    
    # Add capability nodes for each unique agent type
    cap_counter = 1
    for agent_type, capabilities in unique_agents.items():
        if capabilities:
            cap_id = f"CAP{cap_counter}"
            top_caps = capabilities[:3]  # Show top 3 capabilities
            cap_text = "\\n".join(top_caps)
            mermaid_code.append(f"    {cap_id}[\"{agent_type} Capabilities:\\n{cap_text}\"]")
            cap_counter += 1
    
    # Add monitoring and error handling
    mermaid_code.extend([
        "",
        "    %% System Components",
        "    EH[\"🛡️ Error Handler\\nData Validation\\nRetry Logic\\nAlert System\"]",
        "    MON[\"📊 Performance Monitor\\nMetrics Collection\\nResource Usage\\nQuality Metrics\"]",
        "    AI[\"🤖 AI Orchestrator\\nWorkflow Optimization\\nPredictive Analytics\\nSmart Routing\"]"
    ])
    
    # Add styling based on node types
    mermaid_code.extend([
        "",
        "    %% Styling",
        "    classDef inputNode fill:#e8f5e8,stroke:#4caf50,stroke-width:2px",
        "    classDef processNode fill:#e3f2fd,stroke:#2196f3,stroke-width:2px", 
        "    classDef outputNode fill:#fff3e0,stroke:#ff9800,stroke-width:2px",
        "    classDef transformNode fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px",
        "    classDef systemNode fill:#ffebee,stroke:#f44336,stroke-width:2px",
        "    classDef capabilityNode fill:#f5f5f5,stroke:#757575,stroke-width:1px,stroke-dasharray: 5 5",
        ""
    ])
    
    # Classify nodes by type for styling
    input_nodes = []
    process_nodes = []
    output_nodes = []
    transform_nodes = []
    
    for tool_id, node in nodes.items():
        node_id = f"N{tool_id}"
        if node.tool_type in ["Text Input", "Input Data"]:
            input_nodes.append(node_id)
        elif node.tool_type in ["Browse", "Output Data"]:
            output_nodes.append(node_id)
        elif node.tool_type in ["Formula", "Filter", "Join", "Union"]:
            transform_nodes.append(node_id)
        else:
            process_nodes.append(node_id)
    
    # Apply styling
    if input_nodes:
        mermaid_code.append(f"    class {','.join(input_nodes)} inputNode")
    if process_nodes:
        mermaid_code.append(f"    class {','.join(process_nodes)} processNode")
    if output_nodes:
        mermaid_code.append(f"    class {','.join(output_nodes)} outputNode")
    if transform_nodes:
        mermaid_code.append(f"    class {','.join(transform_nodes)} transformNode")
    
    mermaid_code.extend([
        "    class EH,MON,AI systemNode",
        f"    class CAP{',CAP'.join([str(i) for i in range(1, cap_counter)])} capabilityNode" if cap_counter > 1 else ""
    ])
    
    return "\n".join(mermaid_code)

print("Mermaid generation functions defined successfully!")

# ============================================================================
# CELL 7: Report Generation
# ============================================================================

def generate_report(analysis: Dict, nodes: Dict, connections: List, output_file: Path):
    """Generate a detailed markdown report"""
    
    with open(output_file, 'w') as f:
        f.write("# AI-Agent Workflow Analysis Report\n\n")
        
        f.write("## Original Workflow Summary\n")
        f.write(f"- **Total Nodes**: {len(nodes)}\n")
        f.write(f"- **Total Connections**: {len(connections)}\n")
        f.write(f"- **Workflow Type**: Data Processing Pipeline\n\n")
        
        f.write("## AI-Agent Implementation Steps\n\n")
        for i, agent in enumerate(analysis["ai_agents"], 1):
            f.write(f"### Step {i}: {agent.agent_name}\n")
            f.write(f"**Description**: {agent.description}\n\n")
            f.write("**AI Capabilities**:\n")
            for capability in agent.ai_capabilities:
                f.write(f"- {capability}\n")
            f.write(f"\n**Error Handling**: {agent.error_handling}\n")
            f.write(f"**Monitoring**: {agent.monitoring}\n\n")
        
        f.write("## AI-Powered Improvements\n\n")
        for improvement in analysis["improvements"]:
            f.write(f"- {improvement}\n")
        
        f.write("\n## Implementation Recommendations\n\n")
        f.write("1. **Microservices Architecture**: Deploy each AI agent as a separate microservice\n")
        f.write("2. **Event-Driven Processing**: Use message queues for agent communication\n")
        f.write("3. **Containerization**: Use Docker for consistent deployment\n")
        f.write("4. **Monitoring Stack**: Implement comprehensive logging and monitoring\n")
        f.write("5. **CI/CD Pipeline**: Automated testing and deployment\n")

# ============================================================================
# CELL 9.5: Workflow Debugging and Inspection
# ============================================================================

def inspect_workflow(file_path: str):
    """Inspect and debug what's actually in the workflow file"""
    print(f"🔍 Inspecting workflow file: {file_path}")
    print("="*60)
    
    try:
        nodes, connections = parse_workflow_file(file_path)
        
        print(f"📊 Workflow Statistics:")
        print(f"   • Total nodes: {len(nodes)}")
        print(f"   • Total connections: {len(connections)}")
        print()
        
        print(f"🔧 Node Details:")
        for tool_id, node in sorted(nodes.items()):
            print(f"   Node {tool_id}:")
            print(f"      • Type: {node.tool_type}")
            print(f"      • Position: {node.position}")
            if node.annotation:
                print(f"      • Annotation: {node.annotation}")
            print()
        
        print(f"🔗 Connection Details:")
        for i, conn in enumerate(connections, 1):
            source_node = nodes.get(conn.source_tool_id)
            target_node = nodes.get(conn.target_tool_id)
            print(f"   Connection {i}:")
            print(f"      • From: {source_node.tool_type if source_node else 'Unknown'} (ID: {conn.source_tool_id})")
            print(f"      • To: {target_node.tool_type if target_node else 'Unknown'} (ID: {conn.target_tool_id})")
            if conn.source_connection != "Output" or conn.target_connection != "Input":
                print(f"      • Via: {conn.source_connection} → {conn.target_connection}")
            print()
        
        print(f"🤖 AI Agent Mapping Preview:")
        for tool_id, node in sorted(nodes.items()):
            agent_name = map_node_to_agent(node.tool_type)
            capabilities = get_agent_capabilities(node.tool_type, node.configuration)
            print(f"   {node.tool_type} → {agent_name}")
            print(f"      • Capabilities: {', '.join(capabilities[:3])}...")
            print()
            
        return nodes, connections
        
    except Exception as e:
        print(f"❌ Error inspecting workflow: {e}")
        return None, None

print("Workflow inspection function defined!")

# ============================================================================
# CELL 8: Sample Workflow Generator for Testing
# ============================================================================

def create_sample_workflow():
    """Create a sample Alteryx workflow XML for testing"""
    sample_xml = '''<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2018.3">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.TextInput.TextInput">
        <Position x="138" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <NumRows value="7" />
          <Fields>
            <Field name="S no." />
            <Field name="SCP" />
            <Field name="DaysLastYear-Maternity" />
          </Fields>
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>Sample Input Data</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
    <Node ToolID="2">
      <GuiSettings Plugin="AlteryxSpatialPluginsGui.Summarize.Summarize">
        <Position x="174" y="126" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <SummarizeFields>
            <SummarizeField field="SCP" action="Sum" rename="Sum_SCP" />
          </SummarizeFields>
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>Calculate Total SCP</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
    <Node ToolID="3">
      <GuiSettings Plugin="AlteryxBasePluginsGui.BrowseV2.BrowseV2">
        <Position x="150" y="198" />
      </GuiSettings>
      <Properties>
        <Configuration />
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>View Results</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
  </Nodes>
  <Connections>
    <Connection>
      <Origin ToolID="1" Connection="Output" />
      <Destination ToolID="2" Connection="Input" />
    </Connection>
    <Connection>
      <Origin ToolID="2" Connection="Output" />
      <Destination ToolID="3" Connection="Input" />
    </Connection>
  </Connections>
</AlteryxDocument>'''
    
    sample_file = "sample_workflow.yxmd"
    with open(sample_file, "w") as f:
        f.write(sample_xml)
    
    print(f"Created {sample_file} for testing")
    return sample_file

# ============================================================================
# CELL 9: Main Analysis Function
# ============================================================================

def analyze_workflow(file_path: str, api_key: str, output_dir: str = "./output", model: str = "gpt-4") -> Dict:
    """Main function to analyze an Alteryx workflow file and generate outputs"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Parse workflow file
    logger.info(f"Parsing workflow file: {file_path}")
    nodes, connections = parse_workflow_file(file_path)
    
    # Analyze with AI
    logger.info("Analyzing workflow with AI...")
    analysis_result = analyze_workflow_with_ai(nodes, connections, api_key, model)
    
    # Generate Mermaid diagram
    logger.info("Generating Mermaid diagram...")
    mermaid_code = generate_mermaid_diagram(
        nodes,  # Pass the actual nodes instead of ai_agents
        connections,
        analysis_result["ai_agents"]
    )
    
    # Save results
    workflow_name = Path(file_path).stem
    
    # Save analysis JSON
    analysis_file = output_path / f"{workflow_name}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump({
            'workflow_file': file_path,
            'analysis': analysis_result,
            'node_count': len(nodes),
            'connection_count': len(connections)
        }, f, indent=2, default=str)
    
    # Save Mermaid diagram
    mermaid_file = output_path / f"{workflow_name}_ai_agent_workflow.mmd"
    with open(mermaid_file, 'w') as f:
        f.write(mermaid_code)
    
    # Save detailed report
    report_file = output_path / f"{workflow_name}_ai_agent_report.md"
    generate_report(analysis_result, nodes, connections, report_file)
    
    logger.info(f"Analysis complete! Files saved to {output_dir}")
    
    return {
        'analysis_file': str(analysis_file),
        'mermaid_file': str(mermaid_file),
        'report_file': str(report_file),
        'analysis': analysis_result,
        'mermaid_code': mermaid_code
    }

print("Main analysis function defined successfully!")
print("\nAll functions loaded! You can now run the analysis.")

# ============================================================================
# CELL 10: Usage Example - Run this cell to analyze a workflow
# ============================================================================

# Configuration - Update these values
OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual API key
WORKFLOW_FILE = "TT.yxmd"  # Update this to match your actual workflow file
OUTPUT_DIR = "./analysis_output"

# Step 1: First, let's inspect your workflow to see what's actually in it
print("🔍 First, let's inspect your TT.yxmd workflow:")
nodes, connections = inspect_workflow(WORKFLOW_FILE)

if nodes and connections:
    print("\n" + "="*60)
    print("🚀 Now running full analysis...")
    
    # Step 2: Set your API key (you can also get it from environment variable)
    api_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

    if api_key == "your-api-key-here" or not api_key:
        print("⚠️ Please set your OpenAI API key!")
        print("Either update OPENAI_API_KEY variable above or set environment variable:")
        print("export OPENAI_API_KEY='your-actual-key'")
        print("\nFor now, running with fallback analysis (no AI)...")
        
        # Run analysis without API
        try:
            result = analyze_workflow(
                file_path=WORKFLOW_FILE,
                api_key="fallback",  # This will trigger fallback mode
                output_dir=OUTPUT_DIR
            )
            
            print("✅ Analysis completed successfully!")
            print(f"Generated files:")
            print(f"  - Analysis: {result['analysis_file']}")
            print(f"  - Mermaid diagram: {result['mermaid_file']}")
            print(f"  - Report: {result['report_file']}")
            
            # Display the Mermaid code
            print(f"\n📊 Mermaid Diagram Code:")
            print("="*60)
            print(result['mermaid_code'])
            print("="*60)
            
            # Display AI agents found
            agents = result['analysis']['ai_agents']
            print(f"\n🤖 AI Agents Identified ({len(agents)} total):")
            for i, agent in enumerate(agents, 1):
                print(f"  {i}. {agent.agent_name}")
                print(f"     Description: {agent.description}")
                print(f"     Capabilities: {', '.join(agent.ai_capabilities[:3])}...")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
    
    else:
        # Run with full AI analysis
        try:
            result = analyze_workflow(
                file_path=WORKFLOW_FILE,
                api_key=api_key,
                output_dir=OUTPUT_DIR
            )
            
            print("✅ Analysis completed successfully!")
            print(f"Generated files:")
            print(f"  - Analysis: {result['analysis_file']}")
            print(f"  - Mermaid diagram: {result['mermaid_file']}")
            print(f"  - Report: {result['report_file']}")
            
            # Display the Mermaid code
            print(f"\n📊 Mermaid Diagram Code:")
            print("="*60)
            print(result['mermaid_code'])
            print("="*60)
            
            # Display AI agents found
            agents = result['analysis']['ai_agents']
            print(f"\n🤖 AI Agents Identified ({len(agents)} total):")
            for i, agent in enumerate(agents, 1):
                print(f"  {i}. {agent.agent_name}")
                print(f"     Description: {agent.description}")
                print(f"     Capabilities: {', '.join(agent.ai_capabilities[:3])}...")
                
        except Exception as e:
            print(f"❌ Error during analysis: {e}")

print("\n💡 To run the analysis:")
print("1. Update WORKFLOW_FILE variable above to point to your .yxmd file")
print("2. Optionally set your OPENAI_API_KEY for AI-powered analysis")
print("3. Run this cell!")
print("4. Check the generated .mmd file and paste it into https://mermaid.live/ to visualize")
