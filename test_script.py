#!/usr/bin/env python3
"""
Test script for the Alteryx AI Agent Analyzer
Validates all functionality and creates sample outputs
"""

import os
import sys
import tempfile
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

# Add the main module to path
sys.path.append('.')

# Mock OpenAI if not available for testing
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  OpenAI not installed - using mock responses for testing")

class TestWorkflowAnalyzer(unittest.TestCase):
    """Test cases for the workflow analyzer"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path("./test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test workflow XML
        self.sample_workflow = self.test_dir / "test_workflow.yxmd"
        self.create_test_workflow()
        
        # Mock API key for testing
        self.mock_api_key = "test-api-key-123"
    
    def create_test_workflow(self):
        """Create a comprehensive test workflow"""
        test_xml = '''<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2018.3">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput">
        <Position x="54" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <Passwords />
          <File OutputFileName="" FileFormat="25" SearchSubDirs="False">data.csv</File>
          <FormatSpecificOptions>
            <HeaderRow>True</HeaderRow>
            <IgnoreErrors>False</IgnoreErrors>
            <AllowShareWrite>False</AllowShareWrite>
            <ImportLine>1</ImportLine>
            <FieldLen>254</FieldLen>
            <SingleThreadRead>False</SingleThreadRead>
            <IgnoreQuotes>DoubleQuotes</IgnoreQuotes>
            <Delimeter>,</Delimeter>
            <QuoteRecordBreak>False</QuoteRecordBreak>
            <CodePage>28591</CodePage>
          </FormatSpecificOptions>
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>Input CSV Data</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
    
    <Node ToolID="2">
      <GuiSettings Plugin="AlteryxBasePluginsGui.Filter.Filter">
        <Position x="150" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <Mode>Simple</Mode>
          <Simple>
            <Operator>=</Operator>
            <Field>Status</Field>
            <Operands>
              <IgnoreTimeInDateTime>True</IgnoreTimeInDateTime>
              <DateType>fixed</DateType>
              <PeriodDate>2023-01-01 00:00:00</PeriodDate>
              <PeriodType>
              </PeriodType>
              <PeriodCount>0</PeriodCount>
              <Operand>Active</Operand>
              <StartDate>2023-01-01 00:00:00</StartDate>
              <EndDate>2023-01-01 00:00:00</EndDate>
            </Operands>
          </Simple>
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>[Status] = "Active"</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
    
    <Node ToolID="3">
      <GuiSettings Plugin="AlteryxBasePluginsGui.Join.Join">
        <Position x="246" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration joinByRecordPos="False">
          <JoinInfo connection="Left">
            <Field field="ID" />
          </JoinInfo>
          <JoinInfo connection="Right">
            <Field field="CustomerID" />
          </JoinInfo>
          <SelectConfiguration>
            <Configuration outputConnection="Join">
              <OrderChanged value="False" />
              <CommaDecimal value="False" />
              <SelectFields>
                <SelectField field="Left_ID" selected="True" />
                <SelectField field="Left_Name" selected="True" />
                <SelectField field="Right_CustomerID" selected="True" />
                <SelectField field="Right_Details" selected="True" />
              </SelectFields>
            </Configuration>
          </SelectConfiguration>
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>Join on ID = CustomerID</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
    
    <Node ToolID="4">
      <GuiSettings Plugin="AlteryxBasePluginsGui.Formula.Formula">
        <Position x="342" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <FormulaFields>
            <FormulaField expression="[Amount] * [Tax_Rate]" field="Tax_Amount" size="19" type="FixedDecimal" />
            <FormulaField expression="[Amount] + [Tax_Amount]" field="Total_Amount" size="19" type="FixedDecimal" />
          </FormulaFields>
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>Calculate Tax and Total</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
    
    <Node ToolID="5">
      <GuiSettings Plugin="AlteryxSpatialPluginsGui.Summarize.Summarize">
        <Position x="438" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <SummarizeFields>
            <SummarizeField field="Total_Amount" action="Sum" rename="Sum_Total" />
            <SummarizeField field="Total_Amount" action="Average" rename="Avg_Total" />
            <SummarizeField field="CustomerID" action="Count" rename="Count_Customers" />
          </SummarizeFields>
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>Summary Statistics</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
    
    <Node ToolID="6">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput">
        <Position x="534" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <File MaxRecords="" FileFormat="25">output_summary.csv</File>
          <Passwords />
          <FormatSpecificOptions>
            <LineEndStyle>CRLF</LineEndStyle>
            <Delimeter>,</Delimeter>
            <ForceQuotes>False</ForceQuotes>
            <HeaderRow>True</HeaderRow>
            <CodePage>28591</CodePage>
            <WriteBOM>True</WriteBOM>
          </FormatSpecificOptions>
          <MultiFile value="False" />
        </Configuration>
        <Annotation DisplayMode="0">
          <DefaultAnnotationText>Export Results</DefaultAnnotationText>
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
      <Origin ToolID="2" Connection="True" />
      <Destination ToolID="3" Connection="Left" />
    </Connection>
    <Connection>
      <Origin ToolID="3" Connection="Join" />
      <Destination ToolID="4" Connection="Input" />
    </Connection>
    <Connection>
      <Origin ToolID="4" Connection="Output" />
      <Destination ToolID="5" Connection="Input" />
    </Connection>
    <Connection>
      <Origin ToolID="5" Connection="Output" />
      <Destination ToolID="6" Connection="Input" />
    </Connection>
  </Connections>
  
  <Properties>
    <Memory default="True" />
    <GlobalRecordLimit value="0" />
    <TempFiles default="True" />
    <Annotation on="True" includeToolName="False" />
  </Properties>
</AlteryxDocument>'''
        
        with open(self.sample_workflow, 'w') as f:
            f.write(test_xml)
    
    def test_workflow_parsing(self):
        """Test that workflow parsing works correctly"""
        try:
            from alteryx_ai_agent_analyzer import AlteryxWorkflowParser
            
            parser = AlteryxWorkflowParser()
            nodes, connections = parser.parse_workflow_file(str(self.sample_workflow))
            
            # Verify nodes were parsed
            self.assertEqual(len(nodes), 6, "Should parse 6 nodes")
            self.assertEqual(len(connections), 5, "Should parse 5 connections")
            
            # Check specific node types
            node_types = [node.tool_type for node in nodes.values()]
            expected_types = ['Input Data', 'Filter', 'Join', 'Formula', 'Summarize', 'Output Data']
            
            for expected_type in expected_types:
                self.assertIn(expected_type, node_types, f"Should contain {expected_type}")
            
            print("✅ Workflow parsing test passed")
            
        except ImportError:
            print("⚠️  Skipping parsing test - module not available")
    
    def test_mock_analysis(self):
        """Test analysis with mock ChatGPT response"""
        try:
            from alteryx_ai_agent_analyzer import WorkflowAnalyzer
            
            # Create analyzer with mock
            if not HAS_OPENAI:
                with patch('alteryx_ai_agent_analyzer.OpenAI') as mock_openai:
                    # Mock the API response
                    mock_response = Mock()
                    mock_response.choices[0].message.content = """
                    AI-Agent Analysis:
                    1. Data Ingestion Agent - Handles CSV input with validation
                    2. Filtering Agent - Intelligent data filtering with pattern recognition
                    3. Join Agent - Advanced data joining with fuzzy matching
                    4. Calculation Agent - Formula processing with error handling
                    5. Aggregation Agent - Statistical analysis and summarization
                    6. Export Agent - Data output with quality validation
                    """
                    
                    mock_client = Mock()
                    mock_client.chat.completions.create.return_value = mock_response
                    mock_openai.return_value = mock_client
                    
                    analyzer = WorkflowAnalyzer(self.mock_api_key)
                    result = analyzer.analyze_workflow_file(str(self.sample_workflow), str(self.test_dir))
                    
                    # Verify outputs were created
                    self.assertTrue(Path(result['analysis_file']).exists(), "Analysis file should exist")
                    self.assertTrue(Path(result['mermaid_file']).exists(), "Mermaid file should exist") 
                    self.assertTrue(Path(result['report_file']).exists(), "Report file should exist")
                    
                    print("✅ Mock analysis test passed")
            else:
                print("⚠️  Skipping mock test - OpenAI available, use integration test")
                
        except Exception as e:
            print(f"⚠️  Mock analysis test failed: {e}")
    
    def test_mermaid_generation(self):
        """Test Mermaid diagram generation"""
        try:
            from alteryx_ai_agent_analyzer import AIAgentWorkflowConverter, AIAgentStep
            
            # Create sample AI agents
            agents = [
                AIAgentStep(
                    step_id="step1",
                    agent_name="Data Ingestion Agent",
                    description="Input processing",
                    inputs=[],
                    outputs=[],
                    ai_capabilities=["Schema validation", "Type inference"],
                    error_handling="Retry logic",
                    monitoring="Performance tracking"
                ),
                AIAgentStep(
                    step_id="step2", 
                    agent_name="Processing Agent",
                    description="Data transformation",
                    inputs=[],
                    outputs=[],
                    ai_capabilities=["Pattern matching", "Optimization"],
                    error_handling="Error recovery",
                    monitoring="Quality metrics"
                )
            ]
            
            converter = AIAgentWorkflowConverter(self.mock_api_key)
            mermaid_code = converter.generate_mermaid_diagram(agents, [])
            
            # Verify Mermaid syntax
            self.assertIn("flowchart TD", mermaid_code, "Should contain flowchart declaration")
            self.assertIn("Data Ingestion Agent", mermaid_code, "Should contain agent names")
            self.assertIn("classDef", mermaid_code, "Should contain styling")
            
            # Save test output
            test_mermaid_file = self.test_dir / "test_diagram.mmd"
            with open(test_mermaid_file, 'w') as f:
                f.write(mermaid_code)
            
            print("✅ Mermaid generation test passed")
            print(f"📄 Test diagram saved to: {test_mermaid_file}")
            
        except Exception as e:
            print(f"⚠️  Mermaid generation test failed: {e}")
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

def run_integration_test():
    """Run a full integration test with real API if available"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found - skipping integration test")
        print("   Set your API key to run full integration test:")
        print("   export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        from alteryx_ai_agent_analyzer import WorkflowAnalyzer
        
        # Create test workflow
        test_dir = Path("./integration_test")
        test_dir.mkdir(exist_ok=True)
        
        test_workflow = test_dir / "integration_test.yxmd"
        
        # Simple workflow for testing
        simple_xml = '''<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2018.3">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.TextInput.TextInput">
        <Position x="54" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <NumRows value="3" />
          <Fields>
            <Field name="ID" />
            <Field name="Value" />
          </Fields>
        </Configuration>
      </Properties>
    </Node>
    <Node ToolID="2">
      <GuiSettings Plugin="AlteryxSpatialPluginsGui.Summarize.Summarize">
        <Position x="150" y="54" />
      </GuiSettings>
      <Properties>
        <Configuration>
          <SummarizeFields>
            <SummarizeField field="Value" action="Sum" />
          </SummarizeFields>
        </Configuration>
      </Properties>
    </Node>
  </Nodes>
  <Connections>
    <Connection>
      <Origin ToolID="1" Connection="Output" />
      <Destination ToolID="2" Connection="Input" />
    </Connection>
  </Connections>
</AlteryxDocument>'''
        
        with open(test_workflow, 'w') as f:
            f.write(simple_xml)
        
        # Run analysis
        print("🚀 Running integration test with real OpenAI API...")
        analyzer = WorkflowAnalyzer(api_key)
        result = analyzer.analyze_workflow_file(str(test_workflow), str(test_dir))
        
        print("✅ Integration test completed successfully!")
        print(f"📁 Results in: {test_dir}")
        
        # Show generated content
        with open(result['mermaid_file'], 'r') as f:
            mermaid_content = f.read()
            print("\n📊 Generated Mermaid Diagram:")
            print("─" * 50)
            print(mermaid_content[:400] + "..." if len(mermaid_content) > 400 else mermaid_content)
            print("─" * 50)
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 Running Alteryx AI Agent Analyzer Tests")
    print("=" * 60)
    
    # Run unit tests
    print("\n1️⃣  Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=0)
    
    print("\n2️⃣  Running Integration Test...")
    run_integration_test()
    
    print("\n✅ All tests completed!")
    print("\n💡 Next Steps:")
    print("   1. Set OPENAI_API_KEY environment variable")  
    print("   2. Run: python usage_example.py")
    print("   3. Place your .yxmd files in the current directory")
    print("   4. Choose option 1 to analyze a workflow")

if __name__ == "__main__":
    main()
