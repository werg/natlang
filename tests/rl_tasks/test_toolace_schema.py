import json
import shutil
import subprocess
from pathlib import Path

from scripts.rl_tasks.sources.tools import SERVICE_HEAD, _toolace_tools, schema_ts_type, service


def sequence_tool():
    return {'name':'sequence_alignment', 'description':'Align supplied sequences.', 'parameters':{
        'type':'dict', 'properties':{'sequences':{'type':'array', 'items':{
            'type':'dict', 'properties':{
                'sequence_id':{'type':'string', 'description':'Identifier.'},
                'sequence_data':{'type':'string'},
                'sequence_type':{'type':'string', 'enum':['DNA','RNA','Protein']},
            }, 'required':['sequence_id','sequence_data','sequence_type']},
            'description':'List of sequences.'}}, 'required':['sequences']}}


def test_toolace_preserves_nested_contract_in_executable_service(tmp_path):
    original = sequence_tool()
    tools = _toolace_tools('Available functions: '+json.dumps([original]))
    code, _ = service(tools)
    assert tools[0]['json_schema'] == original['parameters']
    assert tools[0]['parameters']['sequences']['json_schema'] == original['parameters']['properties']['sequences']
    assert '"sequence_id": string;' in code
    assert '"sequence_type": ("DNA" | "RNA" | "Protein");' in code
    assert 'Parameters JSON Schema: '+json.dumps(original['parameters']) in code
    # Check the actual generated function signature: a valid request compiles,
    # invented object keys and enum values are rejected before any effect.
    root = Path(__file__).resolve().parents[2]
    compiler = next(p for p in (root/'node_modules/typescript', root/'ts-host/node_modules/typescript') if p.exists())
    source = tmp_path/'contract.ts'
    body = 'declare function receipt(name:string,spec:unknown,args:unknown):string;\n'+code.removeprefix(SERVICE_HEAD)
    valid = '\nsequence_alignment({sequences:[{sequence_id:"A",sequence_data:"ACTG",sequence_type:"DNA"}]});'
    wrong = '\nsequence_alignment({sequences:[{id:"A",sequence_data:"ACTG",sequence_type:"DNA"}]});'
    wrong_enum = '\nsequence_alignment({sequences:[{sequence_id:"A",sequence_data:"ACTG",sequence_type:"dna"}]});'
    driver = '''const ts=require(process.argv[1]);
const program=ts.createProgram([process.argv[2]],{noEmit:true,strict:true,types:[],target:ts.ScriptTarget.ES2022});
console.log(JSON.stringify(ts.getPreEmitDiagnostics(program).map(d=>d.code)));'''
    for call, expected in ((valid,[]),(wrong,[2353]),(wrong_enum,[2820])):
        source.write_text(body+call)
        result = subprocess.run([shutil.which('node'),'-e',driver,str(compiler),str(source)],
                                capture_output=True,text=True,check=True,timeout=30)
        errors=json.loads(result.stdout)
        assert errors == expected, errors


def test_schema_unions_optional_fields_and_unexpressed_constraints_are_retained():
    assert schema_ts_type({'type':['string','null']}) == '(string | null)'
    assert schema_ts_type({'anyOf':[{'type':'string'},{'type':'integer'}]}) == '(string | number)'
    assert '"label"?: string;' in schema_ts_type({'type':'object','properties':{'label':{'type':'string'}}})
    original=sequence_tool()
    original['parameters']['properties']['sequences']['minItems']=2
    tools=_toolace_tools('Functions: '+json.dumps([original]))
    code,_=service(tools)
    assert '"minItems": 2' in code


def test_legacy_xlam_parameter_contract_is_preserved():
    code,_=service([{'name':'count','parameters':{'limit':{'type':'int, optional','default':10}}}])
    assert '"limit"?: number;' in code
    assert '"kind": "number", "default": 10, "optional": true' in code

