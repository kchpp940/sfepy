# This __init__.py is intentionally kept minimal.  The availability of
# the compiled C helpers it would normally expose is probed lazily by
# ``sfepy.base.deps.dep_manager`` on first use, so that every missing
# extension yields a consistent, actionable error message (see
# ``sfepy.base.deps`` for the canonical list of install hints.
