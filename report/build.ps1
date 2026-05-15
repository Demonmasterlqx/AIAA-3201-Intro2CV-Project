$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
