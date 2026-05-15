"""Compatibility shims for the official ZSON reproduction environment."""

import importlib

import distutils


if not hasattr(distutils, "version"):
    distutils.version = importlib.import_module("distutils.version")
