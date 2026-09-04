$Out = "FINAL_REPRODUCIBILITY_RECORD.txt"

"============================================================" | Set-Content $Out
"FC-HMARL FINAL REPRODUCIBILITY RECORD" | Add-Content $Out
"Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $Out
"Project: $((Get-Location).Path)" | Add-Content $Out
"============================================================" | Add-Content $Out

"`n[PYTHON]" | Add-Content $Out
python --version 2>&1 | Add-Content $Out

"`n[PIP]" | Add-Content $Out
python -m pip --version 2>&1 | Add-Content $Out

"`n[PYTORCH]" | Add-Content $Out
python -c "import torch; print('Version:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('Device count:',torch.cuda.device_count())" 2>&1 | Add-Content $Out

"`n[NUMPY / PANDAS / SCIPY / MATPLOTLIB]" | Add-Content $Out
python -c "import numpy,pandas,scipy,matplotlib; print('numpy',numpy.__version__); print('pandas',pandas.__version__); print('scipy',scipy.__version__); print('matplotlib',matplotlib.__version__)" 2>&1 | Add-Content $Out

"`n[PYTEST]" | Add-Content $Out
python -m pytest --version 2>&1 | Add-Content $Out

"`n[FINAL SOFTWARE VERIFICATION]" | Add-Content $Out
"993 tests passed, 22 warnings." | Add-Content $Out

"`n[FINAL TRAINING]" | Add-Content $Out
"Final controller: FC-HMARL V3" | Add-Content $Out
"Training episodes: 5000" | Add-Content $Out
"Final checkpoint selection: validation-based" | Add-Content $Out
"Frozen checkpoint: episode 200" | Add-Content $Out

"`n[FINAL EVALUATION PROTOCOL]" | Add-Content $Out
"TRAIN data used for model training." | Add-Content $Out
"VALIDATION data used for model/checkpoint selection." | Add-Content $Out
"TEST data reserved for final locked evaluation." | Add-Content $Out
"No policy learning or checkpoint re-ranking performed during final TEST evaluation." | Add-Content $Out

"`n[KEY RECORD FILES]" | Add-Content $Out
"DEVELOPMENT_HISTORY.md" | Add-Content $Out
"FINAL_SUBMISSION_MANIFEST.txt" | Add-Content $Out
"FINAL_TEST_VERIFICATION.txt" | Add-Content $Out
"final_project_structure.txt" | Add-Content $Out

"`n============================================================" | Add-Content $Out
"END OF REPRODUCIBILITY RECORD" | Add-Content $Out
"============================================================" | Add-Content $Out

Write-Host "Created: $Out"
