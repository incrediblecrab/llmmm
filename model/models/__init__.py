"""Model compartments.

One package per model *family*. A compartment is self-contained: it owns its
training code, its default hyperparameters and its model card, and it declares
itself to the core library through ``@register``. Nothing outside a compartment
imports its internals, and no central list names it — discovery walks this
directory, so a new family is added by creating a folder.

See ``models/README.md`` for the checklist.
"""
