"""
Allomorph CLI - List Subcommand
Displays tabular catalog overviews of configured instruments, master voices, and tone packs.
"""

from cyclopts import App

from allomorph.cli.console import render_table
from allomorph.cli.targets import get_available_pack_ids
from allomorph.config.instruments import INSTRUMENTS, partition_instrument_bundles
from allomorph.pipeline.pack import PACKS_DIR

list_app = App(
    name="list", help="Inspect catalog instruments, master target voices, and tone packs."
)


@list_app.command(name="instruments")
def list_instruments() -> None:
    """Lists all configured source and target instruments in the catalog."""
    rows = []
    for inst_id, inst in sorted(INSTRUMENTS.items()):
        pickup_names = ", ".join(inst.pickups.keys())
        voicings_count = len(inst.voicings)
        rows.append(
            [
                inst_id,
                inst.name,
                f'{inst.scale_length_in:.1f}"',
                pickup_names,
                str(voicings_count),
            ]
        )
    render_table(
        "Configured Instruments",
        headers=["Instrument ID", "Display Name", "Scale Length", "Pickups", "Voicings"],
        rows=rows,
    )


@list_app.command(name="voices")
def list_voices() -> None:
    """Lists all configured instrument voicings and tone models."""
    rows = []
    for inst_id, inst in sorted(INSTRUMENTS.items()):
        for v_id, v_cfg in sorted(inst.voicings.items()):
            rows.append(
                [
                    f"{inst_id}:{v_id}",
                    v_cfg.tone_name or v_cfg.name or v_id,
                    inst.name,
                    v_cfg.harness,
                ]
            )
    render_table(
        "Instrument Voicings",
        headers=["Voicing Key", "Tone Name", "Instrument", "Harness"],
        rows=rows,
    )


@list_app.command(name="packs")
def list_packs() -> None:
    """Lists all Tone3000 upload packs and bundle partitions."""
    rows = []
    pack_ids = get_available_pack_ids()
    for pid in pack_ids:
        inst = INSTRUMENTS.get(pid)
        inst_name = inst.name if inst else pid
        bundles_str = "N/A"
        stems_count = 0
        if inst:
            bundles = partition_instrument_bundles(inst)
            bundles_str = ", ".join(bundles.keys())
            stems_count = sum(len(b.targets) for b in bundles.values())

        p_dir = PACKS_DIR / pid
        status = "Exported" if (p_dir / "manifest.json").exists() else "Configured"

        rows.append(
            [
                pid,
                inst_name,
                bundles_str,
                str(stems_count),
                status,
            ]
        )
    render_table(
        "Tone3000 Tone Packs",
        headers=["Pack ID", "Instrument", "Pickup Bundles", "Stems Count", "Status"],
        rows=rows,
    )


@list_app.default
def default_list() -> None:
    """Default handler when running 'allomorph list': displays instruments, voices, and packs."""
    list_instruments()
    print()
    list_voices()
    print()
    list_packs()
