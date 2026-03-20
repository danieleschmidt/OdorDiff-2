"""VSCode plugin manifest generator for OdorDiff integration."""

from __future__ import annotations

import json
import os
from typing import Dict, Any


class VSCodePluginManifest:
    """Generates VSCode extension manifest and stub TypeScript code.
    
    Produces a package.json manifest and an extension.ts stub that
    registers OdorDiff commands in Visual Studio Code.
    """

    def generate_package_json(
        self,
        name: str,
        version: str,
        description: str,
    ) -> Dict[str, Any]:
        """Generate a VSCode extension package.json manifest.
        
        Args:
            name: Extension name (e.g., "odordiff-vscode").
            version: Semantic version string (e.g., "1.0.0").
            description: Human-readable extension description.
            
        Returns:
            Dictionary representing the package.json manifest.
        """
        return {
            "name": name,
            "displayName": name,
            "description": description,
            "version": version,
            "engines": {
                "vscode": "^1.70.0"
            },
            "categories": [
                "Other",
                "Machine Learning"
            ],
            "keywords": [
                "odordiff",
                "olfaction",
                "molecule",
                "cheminformatics"
            ],
            "activationEvents": [
                "onCommand:odordiff.generate",
                "onCommand:odordiff.score",
                "onCommand:odordiff.checkNovelty"
            ],
            "main": "./out/extension.js",
            "contributes": {
                "commands": [
                    {
                        "command": "odordiff.generate",
                        "title": "OdorDiff: Generate Molecules"
                    },
                    {
                        "command": "odordiff.score",
                        "title": "OdorDiff: Score Synthesizability"
                    },
                    {
                        "command": "odordiff.checkNovelty",
                        "title": "OdorDiff: Check Novelty"
                    }
                ],
                "configuration": {
                    "title": "OdorDiff",
                    "properties": {
                        "odordiff.pythonPath": {
                            "type": "string",
                            "default": "python3",
                            "description": "Path to Python interpreter with OdorDiff installed."
                        },
                        "odordiff.minSynthesizability": {
                            "type": "number",
                            "default": 0.5,
                            "description": "Minimum synthesizability score for generated molecules."
                        },
                        "odordiff.requireNovel": {
                            "type": "boolean",
                            "default": True,
                            "description": "Only return novel molecules."
                        }
                    }
                }
            },
            "scripts": {
                "vscode:prepublish": "npm run compile",
                "compile": "tsc -p ./",
                "watch": "tsc -watch -p ./"
            },
            "devDependencies": {
                "@types/vscode": "^1.70.0",
                "@types/node": "^18.0.0",
                "typescript": "^5.0.0"
            }
        }

    def generate_extension_stub(self) -> str:
        """Generate TypeScript extension stub code.
        
        Returns:
            TypeScript source code as a string.
        """
        return '''\
import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export function activate(context: vscode.ExtensionContext) {
    console.log('OdorDiff extension activated');

    const config = vscode.workspace.getConfiguration('odordiff');

    // Command: Generate molecules from text description
    const generateCmd = vscode.commands.registerCommand(
        'odordiff.generate',
        async () => {
            const prompt = await vscode.window.showInputBox({
                prompt: 'Enter olfactory description (e.g., "fresh citrus")',
                placeHolder: 'fresh citrus'
            });
            if (!prompt) return;

            const python = config.get<string>('pythonPath', 'python3');
            const minSA = config.get<number>('minSynthesizability', 0.5);
            const requireNovel = config.get<boolean>('requireNovel', true);

            vscode.window.withProgress(
                { location: vscode.ProgressLocation.Notification, title: 'OdorDiff: Generating...' },
                async () => {
                    try {
                        const cmd = `${python} -m odordiff generate "${prompt}"`;
                        const { stdout } = await execAsync(cmd);
                        vscode.window.showInformationMessage(`Generated molecules:\\n${stdout}`);
                    } catch (err: any) {
                        vscode.window.showErrorMessage(`OdorDiff error: ${err.message}`);
                    }
                }
            );
        }
    );

    // Command: Score synthesizability
    const scoreCmd = vscode.commands.registerCommand(
        'odordiff.score',
        async () => {
            const smiles = await vscode.window.showInputBox({
                prompt: 'Enter SMILES string to score',
                placeHolder: 'CC(=O)O'
            });
            if (!smiles) return;

            const python = config.get<string>('pythonPath', 'python3');
            try {
                const { stdout } = await execAsync(
                    `${python} -m odordiff score "${smiles}"`
                );
                vscode.window.showInformationMessage(`SA Score: ${stdout.trim()}`);
            } catch (err: any) {
                vscode.window.showErrorMessage(`OdorDiff error: ${err.message}`);
            }
        }
    );

    // Command: Check novelty
    const noveltyCmd = vscode.commands.registerCommand(
        'odordiff.checkNovelty',
        async () => {
            const smiles = await vscode.window.showInputBox({
                prompt: 'Enter SMILES string to check novelty',
                placeHolder: 'CC(=O)O'
            });
            if (!smiles) return;

            const python = config.get<string>('pythonPath', 'python3');
            try {
                const { stdout } = await execAsync(
                    `${python} -m odordiff check-novelty "${smiles}"`
                );
                vscode.window.showInformationMessage(`Novelty: ${stdout.trim()}`);
            } catch (err: any) {
                vscode.window.showErrorMessage(`OdorDiff error: ${err.message}`);
            }
        }
    );

    context.subscriptions.push(generateCmd, scoreCmd, noveltyCmd);
}

export function deactivate() {
    console.log('OdorDiff extension deactivated');
}
'''

    def save(self, output_dir: str) -> None:
        """Save the VSCode extension files to a directory.
        
        Creates:
            - package.json: Extension manifest
            - extension.ts: TypeScript stub
            
        Args:
            output_dir: Directory to write files to. Created if it doesn't exist.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Generate with default values
        pkg = self.generate_package_json(
            name="odordiff-vscode",
            version="1.0.0",
            description="OdorDiff olfactory molecule generation in VSCode",
        )
        pkg_path = os.path.join(output_dir, "package.json")
        with open(pkg_path, "w", encoding="utf-8") as f:
            json.dump(pkg, f, indent=2)

        ts_path = os.path.join(output_dir, "extension.ts")
        with open(ts_path, "w", encoding="utf-8") as f:
            f.write(self.generate_extension_stub())

    def __repr__(self) -> str:
        return "VSCodePluginManifest()"
