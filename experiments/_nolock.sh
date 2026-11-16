#!/bin/bash
# Default GPU-lock wrapper: no lock. Usage: _nolock.sh "description" command...
shift
exec "$@"
