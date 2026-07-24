# ruff: noqa: E501
import os
import subprocess
import uuid
from typing import Any, Dict

import yaml

from agent.tools.base import BaseTool


class CodeSandboxTool(BaseTool):
    """
    Executes Python code in an isolated, network-disabled environment.
    Use this tool to safely run scikit-learn snippets to verify runtime behavior, 
    shapes, or state.
    """
    
    @property
    def name(self) -> str:
        return "run_code_sandbox"
        
    @property
    def description(self) -> str:
        return (
            "Executes arbitrary Python code in an isolated, secure Docker sandbox with no network access. "
            "Use this to verify code behavior, array shapes, or math. You must provide a complete, "
            "runnable Python script. The tool returns the standard output (stdout) and standard error (stderr)."
        )

    def __init__(self, config_path: str = "configs/sandbox.yaml"):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["sandbox"]
            self.cpu_cores = config["resources"]["cpu_cores"]
            self.memory_limit = config["resources"]["memory_limit"]
            self.exec_timeout = config["timeouts"]["execution_timeout_seconds"]
            self.user = config["security"]["run_as_user"]
            self.network_disabled = config["security"]["network_disabled"]
        
        self.workspace_dir = os.path.abspath("sandbox_workspace")
        os.makedirs(self.workspace_dir, exist_ok=True)

    def execute(self, args: Dict[str, Any]) -> str:
        code = args.get("code")
        if not code:
            return "Error: Missing 'code' parameter."
            
        script_id = str(uuid.uuid4())
        script_filename = f"script_{script_id}.py"
        script_path = os.path.join(self.workspace_dir, script_filename)
        container_name = f"sandbox_{script_id}"
        
        # Write the agent's code to the scratch directory
        with open(script_path, "w") as f:
            f.write(code)
            
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            f"--memory={self.memory_limit}",
            f"--cpus={self.cpu_cores}",
            f"--user={self.user}",
            "-v", f"{self.workspace_dir}:/sandbox",
            "-w", "/sandbox",
        ]
        
        if self.network_disabled:
            docker_cmd.extend(["--network", "none"])
            
        docker_cmd.extend(["waypoint-sandbox:latest", "python", f"/sandbox/{script_filename}"])
        
        try:
            # We enforce the wall-clock timeout at the subprocess level
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.exec_timeout
            )
            
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
                
            if result.returncode != 0:
                output = f"Execution failed (Exit Code {result.returncode}).\n\n{output}"
                
            return output.strip() if output else "Execution completed successfully with no output."
            
        except subprocess.TimeoutExpired:
            # If the timeout hits, the subprocess is killed, but the docker container
            # might still be running in the background. We must forcefully kill it.
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
            return f"Error: Execution exceeded the {self.exec_timeout}-second timeout and was killed."
            
        except Exception as e:
            return f"Error during sandbox execution: {str(e)}"
            
        finally:
            # Always clean up the script so the scratch dir doesn't fill up
            if os.path.exists(script_path):
                os.remove(script_path)

    def get_function_declaration(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "code": {
                        "type": "STRING",
                        "description": "The complete, runnable Python code script to execute. Must include necessary imports (e.g., import numpy as np)."
                    }
                },
                "required": ["code"]
            }
        }
