"""Visual Memory Assistant vision service.

Consumes the Media Gateway's video relay, decides what may be claimed about
an object's location using the interaction/rest state machine
(`domain/stability.py`), and posts confirmed candidates to the Memory
Service. See `docs/06-Data-Contract.md`'s "Candidate verification boundary"
for the contract this produces against.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
