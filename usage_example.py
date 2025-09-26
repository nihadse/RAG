#!/usr/bin/env python3
"""
Example usage of the Alteryx Workflow AI-Agent Analyzer
"""

import os
import json
from pathlib import Path

# Import our analyzer
from alteryx_ai_agent_analyzer import WorkflowAnalyzer

def analyze_sample_workflow():
    """Example function showing how to use the analyzer"""
    
    # Set up your OpenAI API key
    # You can get this from: https://platform.openai.com/api-keys
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Initialize the analyzer
    analyzer = WorkflowAnalyzer(api_key)
    
    # Path to your Alteryx workflow file
    workflow_file = "New Workflow3.yxmd"  # Replace with your workflow file
    
    # Output directory for results
    output_dir = "./ai_agent_analysis"
    
    try:
        # Analyze the workflow
        print("🔍 Analyzing Alteryx workflow...")
        result = analyzer.analyze_workflow_file(workflow_file, output_dir)
        
        # Display results
        print("✅ Analysis completed successfully!")
        print(f"\n📊 Generated Files:")
        print(f"   📄 Analysis JSON: {result['analysis_file']}")
        print(f"   🎨 Mermaid Diagram: {result['mermaid_file']}")
        print(f"   📖 Detailed Report: {result['report_file']}")
        
        # Show a preview of the AI agents identified
        agents = result['analysis']['ai_agents']
        print(f"\n🤖 AI Agents Identified ({len(agents)} total):")
        for i, agent in enumerate(agents, 1):
            print(f"   {i}. {agent.agent_name}")
            print(f"      └─ {agent.description}")
        
        # Show the Mermaid diagram content
        print(f"\n🎨 Mermaid Diagram Preview:")
        with open(result['mermaid_file'], 'r') as f:
            mermaid_content = f.read()
            print("─" * 50)
            print(mermaid_content[:500] + "..." if len(mermaid_content) > 500 else mermaid_content)
            print("─" * 50)
        
        print(f"\n💡 To visualize the Mermaid diagram:")
        print(f"   1. Copy the content from: {result['mermaid_file']}")
        print(f"   2. Paste it into: https://mermaid.live/")
        print(f"   3. Or use VS Code with Mermaid extension")
        
    except FileNotFoundError:
        print(f"❌ Workflow file not found: {workflow_file}")
        print("Please ensure the file exists and the path is correct.")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

def create_sample_workflow_xml():
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
    
    with open("sample_workflow.yxmd", "w") as f:
        f.write(sample_xml)
    
    print("✅ Created sample_workflow.yxmd for testing")

def batch_analyze_workflows():
    """Analyze multiple workflow files in a directory"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    analyzer = WorkflowAnalyzer(api_key)
    
    # Find all .yxmd files in current directory
    workflow_files = list(Path(".").glob("*.yxmd"))
    
    if not workflow_files:
        print("No .yxmd files found in current directory")
        return
    
    print(f"Found {len(workflow_files)} workflow files to analyze:")
    for file in workflow_files:
        print(f"  - {file}")
    
    # Analyze each workflow
    for workflow_file in workflow_files:
        print(f"\n🔍 Analyzing: {workflow_file}")
        try:
            result = analyzer.analyze_workflow_file(str(workflow_file))
            print(f"✅ Completed: {workflow_file}")
        except Exception as e:
            print(f"❌ Failed: {workflow_file} - {e}")

def main():
    """Main function with menu options"""
    print("🚀 Alteryx to AI-Agent Workflow Analyzer")
    print("=" * 50)
    print("1. Analyze single workflow file")
    print("2. Create sample workflow for testing")
    print("3. Batch analyze all workflows in directory")
    print("4. Exit")
    print("=" * 50)
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            workflow_file = input("Enter path to your .yxmd file: ").strip()
            if not workflow_file:
                print("❌ Please provide a file path")
                continue
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                api_key = input("Enter your OpenAI API key: ").strip()
                if not api_key:
                    print("❌ API key is required")
                    continue
            
            analyzer = WorkflowAnalyzer(api_key)
            
            try:
                print(f"\n🔍 Analyzing: {workflow_file}")
                result = analyzer.analyze_workflow_file(workflow_file)
                
                print("✅ Analysis completed!")
                print(f"📁 Files saved to: ./output/")
                
                # Show preview
                with open(result['mermaid_file'], 'r') as f:
                    content = f.read()
                    print(f"\n📄 Mermaid file preview ({result['mermaid_file']}):")
                    print("─" * 60)
                    print(content[:300] + "..." if len(content) > 300 else content)
                    print("─" * 60)
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            create_sample_workflow_xml()
            print("You can now analyze 'sample_workflow.yxmd' using option 1")
        
        elif choice == "3":
            batch_analyze_workflows()
        
        elif choice == "4":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()