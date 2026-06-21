"""Synthetic corpus — a *fictional* product called "Harbor". No real or customer data.

This is deliberately tiny and made-up so the repo is safe to publish and instant to run.
Swap this list for your own documents (or a loader) without touching anything else.
"""

from .models import Chunk

HARBOR_CHUNKS = [
    Chunk(
        "H1",
        "Harbor Docs — Storage",
        "The Free plan includes 5 GB of storage per user. The Pro plan includes 1 TB "
        "of storage per user. Storage is shared across all devices linked to your account.",
    ),
    Chunk(
        "H2",
        "Harbor Docs — Sharing",
        "You can share any folder with people outside your team by creating a share link. "
        "Share links can be set to view-only or edit access. A folder owner can revoke a "
        "share link at any time.",
    ),
    Chunk(
        "H3",
        "Harbor Docs — Sync",
        "Harbor syncs files automatically across all your devices within a few seconds of a "
        "change. If a device is offline, changes sync the next time it reconnects.",
    ),
    Chunk(
        "H4",
        "Harbor Docs — Apps",
        "Harbor has desktop apps for Windows and macOS, and mobile apps for iOS and Android. "
        "A web client is available at app.harbor.example.",
    ),
    Chunk(
        "H5",
        "Harbor Docs — Versions",
        "Harbor keeps the version history of every file for 30 days on the Free plan and 180 "
        "days on the Pro plan. You can restore any previous version from the file's history panel.",
    ),
]
