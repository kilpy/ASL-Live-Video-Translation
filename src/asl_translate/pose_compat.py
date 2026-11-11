"""One verified code-only migration for the already running ViTPose extraction."""

LEGACY_VITPOSE = {
    "pose_sources": "86a5ae0ab97d4e5ef7f9a9dc689e0ff1a3a4d57ec2738e4e80c826eafcb536e0",
    "features": "919dd27f21d254555e4fda437eed28d388395b41f3d559b0bb5f09bc0d1310ac",
}
ADDED_RTM_SOURCES = {
    "pose_sources": "bbb749f0c01a9e388232062b3d70a95f9111a912b22400f22b39a1309b09e3b0",
    "features": "92f35250b1cfb4d08b8b01351e569483310b5c4060ffe0ba664ee60bd164b61c",
}


def compatible_vitpose_metadata(previous: dict, current: dict) -> bool:
    """Allow only the addition of RTM sources, with all ViTPose settings unchanged."""
    return (previous.get("source") == "vitpose"
            and previous.get("code_sha256") == LEGACY_VITPOSE
            and current.get("code_sha256") == ADDED_RTM_SOURCES
            and {k: v for k, v in previous.items() if k != "code_sha256"}
            == {k: v for k, v in current.items() if k != "code_sha256"})
