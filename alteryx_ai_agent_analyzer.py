#!/usr/bin/env python3
"""
Alteryx Workflow to AI-Agent Analyzer
=====================================

This AI agent analyzes Alteryx workflows (.yxmd files) and converts them into:
1. Step-by-step AI-Agent implementation outlines
2. Mermaid flowchart diagrams representing the AI-Agent workflow

Author: AI Assistant
Date: 2025-09-26
"""

import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import openai
from openai import OpenAI
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class AlteryxWorkflowParser:
    """Parses Alteryx workflow files and extracts structure"""
    
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
        'AlteryxBasePluginsGui.Transpose.Transpose': 'Transpose'
    }

    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.connections: List[WorkflowConnection] = []

    def parse_workflow_file(self, file_path: str) -> Tuple[Dict[str, WorkflowNode], List[WorkflowConnection]]:
        """Parse an Alteryx workflow file and extract nodes and connections"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Parse nodes
            nodes_element = root.find('Nodes')
            if nodes_element is not None:
                for node in nodes_element.findall('Node'):
                    self._parse_node(node)
            
            # Parse connections
            connections_element = root.find('Connections')
            if connections_element is not None:
                for connection in connections_element.findall('Connection'):
                    self._parse_connection(connection)
            
            return self.nodes, self.connections
            
        except ET.ParseError as e:
            logger.error(f"Error parsing XML file: {e}")
            raise
        except FileNotFoundError as e:
            logger.error(f"Workflow file not found: {e}")
            raise

    def _parse_node(self, node_element: ET.Element):
        """Parse a single node from the XML"""
        tool_id = node_element.get('ToolID')
        
        # Get GUI settings
        gui_settings = node_element.find('GuiSettings')
        plugin = gui_settings.get('Plugin') if gui_settings is not None else 'Unknown'
        tool_type = self.TOOL_MAPPINGS.get(plugin, plugin)
        
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
                configuration = self._xml_to_dict(config_elem)
            
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
        
        self.nodes[tool_id] = workflow_node

    def _parse_connection(self, connection_element: ET.Element):
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
            
            self.connections.append(connection)

    def _xml_to_dict(self, element: ET.Element) -> Dict:
        """Convert XML element to dictionary"""
        result = {}
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text
            else:
                result[child.tag] = self._xml_to_dict(child)
        return result

class AIAgentWorkflowConverter:
    """Converts Alteryx workflow to AI-Agent implementation using ChatGPT"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def analyze_workflow(self, nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection]) -> Dict:
        """Analyze workflow and generate AI-Agent implementation plan"""
        
        # Create workflow summary
        workflow_summary = self._create_workflow_summary(nodes, connections)
        
        # Generate AI-Agent analysis using ChatGPT
        prompt = self._create_analysis_prompt(workflow_summary)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in data processing workflows and AI agent architecture. Analyze Alteryx workflows and design AI-agent implementations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_ai_analysis(analysis_text, nodes, connections)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Fallback to rule-based analysis
            return self._fallback_analysis(nodes, connections)

    def generate_mermaid_diagram(self, ai_agents: List[AIAgentStep], connections: List[WorkflowConnection]) -> str:
        """Generate Mermaid flowchart for AI-Agent workflow"""
        
        mermaid_code = []
        mermaid_code.append("flowchart TD")
        mermaid_code.append("    %% AI Agent Workflow Implementation")
        mermaid_code.append("")
        
        # Add nodes
        for i, agent in enumerate(ai_agents):
            node_id = f"A{i+1}"
            mermaid_code.append(f"    {node_id}[{agent.agent_name}]")
            
            # Add capabilities as sub-nodes
            if agent.ai_capabilities:
                for j, capability in enumerate(agent.ai_capabilities):
                    cap_id = f"A{i+1}_{j+1}"
                    mermaid_code.append(f"    {cap_id}[{capability}]")
                    mermaid_code.append(f"    {node_id} --> {cap_id}")
        
        mermaid_code.append("")
        
        # Add connections based on workflow flow
        for i in range(len(ai_agents) - 1):
            mermaid_code.append(f"    A{i+1} --> A{i+2}")
        
        # Add error handling and monitoring
        mermaid_code.append("")
        mermaid_code.append("    %% Error Handling and Monitoring")
        mermaid_code.append("    EH[Error Handler] --> EH1[Data Validation]")
        mermaid_code.append("    EH --> EH2[Retry Logic]")
        mermaid_code.append("    EH --> EH3[Alert System]")
        mermaid_code.append("    ")
        mermaid_code.append("    MON[Performance Monitor] --> MON1[Metrics Collection]")
        mermaid_code.append("    MON --> MON2[Resource Usage]")
        mermaid_code.append("    MON --> MON3[Quality Metrics]")
        
        # Add styling
        mermaid_code.extend([
            "",
            "    %% Styling",
            "    classDef agent fill:#e3f2fd",
            "    classDef capability fill:#f3e5f5",
            "    classDef error fill:#ffebee",
            "    classDef monitor fill:#e8f5e8",
            "",
            "    class " + ",".join([f"A{i+1}" for i in range(len(ai_agents))]) + " agent",
            "    class EH,EH1,EH2,EH3 error",
            "    class MON,MON1,MON2,MON3 monitor"
        ])
        
        return "\n".join(mermaid_code)

    def _create_workflow_summary(self, nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection]) -> str:
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

    def _create_analysis_prompt(self, workflow_summary: str) -> str:
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

    def _parse_ai_analysis(self, analysis_text: str, nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection]) -> Dict:
        """Parse ChatGPT analysis into structured format"""
        
        # This is a simplified parser - in practice, you'd want more sophisticated NLP
        ai_agents = []
        
        # Create agents based on the workflow nodes
        for i, (tool_id, node) in enumerate(nodes.items()):
            agent_name = self._map_node_to_agent(node.tool_type)
            capabilities = self._get_agent_capabilities(node.tool_type, node.configuration)
            
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

    def _fallback_analysis(self, nodes: Dict[str, WorkflowNode], connections: List[WorkflowConnection]) -> Dict:
        """Fallback analysis when API is unavailable"""
        ai_agents = []
        
        for i, (tool_id, node) in enumerate(nodes.items()):
            agent_name = self._map_node_to_agent(node.tool_type)
            capabilities = self._get_agent_capabilities(node.tool_type, node.configuration)
            
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

    def _map_node_to_agent(self, tool_type: str) -> str:
        """Map Alteryx tool types to AI agent names"""
        mapping = {
            'Text Input': 'Data Ingestion Agent',
            'Input Data': 'Data Source Agent',
            'Summarize': 'Aggregation Agent',
            'Append Fields': 'Data Enrichment Agent',
            'Formula': 'Calculation Agent',
            'Browse': 'Output Validation Agent',
            'Output Data': 'Data Export Agent',
            'Filter': 'Data Filtering Agent',
            'Join': 'Data Join Agent',
            'Union': 'Data Merge Agent',
            'Sort': 'Data Sorting Agent',
            'Unique': 'Deduplication Agent',
            'Cross Tab': 'Pivot Agent',
            'Transpose': 'Data Transformation Agent'
        }
        
        return mapping.get(tool_type, f'{tool_type} Agent')

    def _get_agent_capabilities(self, tool_type: str, configuration: Dict) -> List[str]:
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

class WorkflowAnalyzer:
    """Main class to orchestrate the workflow analysis"""
    
    def __init__(self, openai_api_key: str):
        self.parser = AlteryxWorkflowParser()
        self.converter = AIAgentWorkflowConverter(openai_api_key)
    
    def analyze_workflow_file(self, file_path: str, output_dir: str = "./output") -> Dict:
        """Analyze an Alteryx workflow file and generate outputs"""
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Parse workflow file
        logger.info(f"Parsing workflow file: {file_path}")
        nodes, connections = self.parser.parse_workflow_file(file_path)
        
        # Analyze with AI
        logger.info("Analyzing workflow with AI...")
        analysis_result = self.converter.analyze_workflow(nodes, connections)
        
        # Generate Mermaid diagram
        logger.info("Generating Mermaid diagram...")
        mermaid_code = self.converter.generate_mermaid_diagram(
            analysis_result["ai_agents"], 
            connections
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
        self._generate_report(analysis_result, nodes, connections, report_file)
        
        logger.info(f"Analysis complete! Files saved to {output_dir}")
        
        return {
            'analysis_file': str(analysis_file),
            'mermaid_file': str(mermaid_file),
            'report_file': str(report_file),
            'analysis': analysis_result
        }
    
    def _generate_report(self, analysis: Dict, nodes: Dict, connections: List, output_file: Path):
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

def main():
    """Main function to run the workflow analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Alteryx workflows and generate AI-Agent implementations')
    parser.add_argument('workflow_file', help='Path to Alteryx workflow file (.yxmd)')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--output-dir', default='./output', help='Output directory for generated files')
    parser.add_argument('--model', default='gpt-4', help='OpenAI model to use')
    
    args = parser.parse_args()
    
    try:
        analyzer = WorkflowAnalyzer(args.api_key)
        result = analyzer.analyze_workflow_file(args.workflow_file, args.output_dir)
        
        print("Analysis completed successfully!")
        print(f"Generated files:")
        print(f"  - Analysis: {result['analysis_file']}")
        print(f"  - Mermaid diagram: {result['mermaid_file']}")
        print(f"  - Detailed report: {result['report_file']}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
