#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "dispatch_warp_png"

W = 1600
H = 900
BG = "#F7F5EF"
TEXT = "#16202A"
MUTED = "#5A6470"
BLUE = "#7DB7FF"
BLUE_DARK = "#2E6FB8"
TEAL = "#90D7C4"
TEAL_DARK = "#2C7F6E"
ORANGE = "#F8B26A"
ORANGE_DARK = "#BA6C17"
GREEN = "#A7D97A"
GREEN_DARK = "#4A8737"
RED = "#F28E8E"
RED_DARK = "#A34343"
PURPLE = "#C1B0FF"
PURPLE_DARK = "#6A4AC7"
GRAY = "#D8DCE2"
GRAY_DARK = "#7A8796"
WHITE = "#FFFFFF"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(44, bold=True)
FONT_SUBTITLE = load_font(24, bold=True)
FONT_BODY = load_font(21)
FONT_SMALL = load_font(17)
FONT_TINY = load_font(15)


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    return img, draw


def wrapped_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    def wrap_one_line(line: str) -> list[str]:
        if not line:
            return [""]
        out: list[str] = []
        buf = ""
        last_space = -1
        for ch in line:
            candidate = buf + ch
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width:
                buf = candidate
                if ch == " ":
                    last_space = len(buf) - 1
                continue
            if last_space != -1:
                out.append(buf[:last_space].rstrip())
                buf = (buf[last_space + 1:] + ch).lstrip()
                last_space = buf.rfind(" ")
            else:
                if buf:
                    out.append(buf.rstrip())
                buf = ch.lstrip()
                last_space = -1
        if buf:
            out.append(buf.rstrip())
        return out or [""]

    paragraphs = text.split("\n")
    lines: list[str] = []
    for idx, paragraph in enumerate(paragraphs):
        lines.extend(wrap_one_line(paragraph))
        if idx != len(paragraphs) - 1:
            lines.append("")
    return "\n".join(lines)


def multiline_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=TEXT, spacing=6) -> None:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    x = xy[0] - (bbox[2] - bbox[0]) / 2
    y = xy[1] - (bbox[3] - bbox[1]) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=spacing, align="center")


def text_block(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, max_width: int, fill=TEXT) -> None:
    wrapped = wrapped_text(draw, text, font, max_width)
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=5)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str,
    outline: str,
    title_fill: str = TEXT,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=28, fill=fill, outline=outline, width=4)
    draw.rounded_rectangle((x1 + 18, y1 + 18, x2 - 18, y1 + 66), radius=18, fill=WHITE)
    draw.text((x1 + 36, y1 + 28), title, font=FONT_SUBTITLE, fill=title_fill)
    text_block(draw, (x1 + 36, y1 + 92), body, FONT_BODY, x2 - x1 - 72)


def pill(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: str, outline: str, text_fill: str = TEXT):
    draw.rounded_rectangle(box, radius=999, fill=fill, outline=outline, width=3)
    multiline_center(draw, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), text, FONT_SMALL, fill=text_fill, spacing=4)


def token_grid(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    labels: list[str],
    cols: int,
    fill: str,
    outline: str,
    cell_w: int = 84,
    cell_h: int = 42,
    gap: int = 12,
) -> None:
    ox, oy = origin
    for idx, label in enumerate(labels):
        row = idx // cols
        col = idx % cols
        x1 = ox + col * (cell_w + gap)
        y1 = oy + row * (cell_h + gap)
        x2 = x1 + cell_w
        y2 = y1 + cell_h
        draw.rounded_rectangle((x1, y1, x2, y2), radius=12, fill=fill, outline=outline, width=2)
        multiline_center(draw, ((x1 + x2) // 2, (y1 + y2) // 2), label, FONT_SMALL)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int = 8,
    label: str | None = None,
    label_offset: tuple[int, int] = (0, 0),
) -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=width)
    angle = math.atan2(ey - sy, ex - sx)
    head = 20
    wing = math.pi / 7
    p1 = (ex, ey)
    p2 = (ex - head * math.cos(angle - wing), ey - head * math.sin(angle - wing))
    p3 = (ex - head * math.cos(angle + wing), ey - head * math.sin(angle + wing))
    draw.polygon([p1, p2, p3], fill=color)
    if label:
        mx = (sx + ex) // 2 + label_offset[0]
        my = (sy + ey) // 2 + label_offset[1]
        bbox = draw.multiline_textbbox((0, 0), label, font=FONT_SMALL, spacing=4, align="center")
        pad = 10
        box = (mx - (bbox[2] - bbox[0]) // 2 - pad, my - (bbox[3] - bbox[1]) // 2 - pad,
               mx + (bbox[2] - bbox[0]) // 2 + pad, my + (bbox[3] - bbox[1]) // 2 + pad)
        draw.rounded_rectangle(box, radius=14, fill=WHITE, outline=color, width=2)
        multiline_center(draw, (mx, my), label, FONT_SMALL, fill=TEXT, spacing=4)


def banner(draw: ImageDraw.ImageDraw, step: str, title: str, subtitle: str) -> None:
    draw.rounded_rectangle((36, 28, W - 36, 138), radius=32, fill=WHITE, outline=GRAY, width=3)
    pill(draw, (58, 52, 208, 96), step, fill=TEXT, outline=TEXT, text_fill=WHITE)
    draw.text((236, 42), title, font=FONT_TITLE, fill=TEXT)
    draw.text((236, 92), subtitle, font=FONT_SMALL, fill=MUTED)


def footer(draw: ImageDraw.ImageDraw, code_ref: str) -> None:
    draw.text((48, H - 42), code_ref, font=FONT_TINY, fill=MUTED)


def save(img: Image.Image, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    img.save(path)
    return path


def step1() -> Path:
    img, draw = new_canvas()
    banner(draw, "STEP 1", "Scan topk_idx and count local expert hits", "Kernel region: dispatch warp reads token-topk slots and accumulates per-expert counts in shared memory.")
    rounded_box(draw, (70, 210, 480, 760), "Source rank input", "Each token has top-k expert ids. A dispatch warp scans token-topk slots in a grid-stride loop.", BLUE, BLUE_DARK)
    token_grid(draw, (120, 340), ["T0-E2", "T0-E5", "T1-E5", "T1-E1", "T2-E2", "T2-E5"], cols=2, fill=WHITE, outline=BLUE_DARK)
    pill(draw, (140, 620, 410, 680), "warp lanes = token × top-k slots", fill="#E9F3FF", outline=BLUE_DARK)
    rounded_box(draw, (580, 270, 1010, 700), "Dispatch warp", "read_topk_idx() assigns active lanes to valid token-topk pairs and calls process(token_topk_idx, expert_idx).", ORANGE, ORANGE_DARK)
    rounded_box(draw, (1120, 230, 1520, 740), "smem_expert_count", "atomicAdd_block increments a shared counter per expert. This is only CTA-local state for now.", TEAL, TEAL_DARK)
    token_grid(draw, (1185, 380), ["E0:0", "E1:1", "E2:2", "E3:0", "E4:0", "E5:3"], cols=2, fill=WHITE, outline=TEAL_DARK)
    arrow(draw, (480, 470), (580, 470), ORANGE_DARK, label="read_topk_idx()", label_offset=(0, -38))
    arrow(draw, (1010, 470), (1120, 470), TEAL_DARK, label="atomicAdd_block", label_offset=(0, -40))
    footer(draw, "Refs: sm100_fp8_fp4_mega_moe.cuh:363-385")
    return save(img, "dispatch_step_1_scan_and_count.png")


def step2() -> Path:
    img, draw = new_canvas()
    banner(draw, "STEP 2", "Reserve global offsets with 64-bit atomic add", "Each SM publishes its local count and receives a unique destination offset for each expert.")
    rounded_box(draw, (70, 210, 500, 760), "Per-SM local counts", "After the CTA barrier, threads iterate experts and publish smem_expert_count[i] to workspace.expert_send_count[i].", BLUE, BLUE_DARK)
    token_grid(draw, (130, 360), ["SM0 E5=2", "SM1 E5=1", "SM2 E5=3", "SM3 E5=0"], cols=2, fill=WHITE, outline=BLUE_DARK, cell_w=132)
    rounded_box(draw, (575, 250, 1030, 720), "64-bit atomicAdd", "send_value = (1 << 32) | local_count.\nlow32 accumulates token count.\nhigh32 counts finished SM reports.", ORANGE, ORANGE_DARK)
    pill(draw, (650, 530, 960, 590), "old low32 = destination offset for this SM", fill="#FFF0DD", outline=ORANGE_DARK)
    rounded_box(draw, (1110, 220, 1520, 740), "workspace.expert_send_count", "Example after all SMs:\nlow32 = total tokens for expert E5\nhigh32 = number of SMs that reported.", TEAL, TEAL_DARK)
    token_grid(draw, (1185, 425), ["high32", "4", "low32", "6"], cols=2, fill=WHITE, outline=TEAL_DARK, cell_w=90)
    arrow(draw, (500, 480), (575, 480), ORANGE_DARK, label="publish local count")
    arrow(draw, (1030, 480), (1110, 480), TEAL_DARK, label="store accumulated status")
    arrow(draw, (900, 620), (760, 620), PURPLE_DARK, label="returned old low32 -> per-SM offset", label_offset=(0, -36))
    footer(draw, "Refs: sm100_fp8_fp4_mega_moe.cuh:388-395, scheduler/mega_moe.cuh:183-195")
    return save(img, "dispatch_step_2_global_offsets.png")


def step3() -> Path:
    img, draw = new_canvas()
    banner(draw, "STEP 3", "Write src_token_topk_idx into the destination rank workspace", "The source rank does a second scan, but now it writes source indices into the expert queue owned by the destination rank.")
    rounded_box(draw, (70, 210, 470, 760), "Source rank token-topk", "For every valid token_topk_idx, compute dst_rank = expert / experts_per_rank and dst_slot = local offset + CTA-local counter.", BLUE, BLUE_DARK)
    token_grid(draw, (120, 375), ["slot 0 -> T0,k0", "slot 1 -> T1,k0", "slot 2 -> T2,k1"], cols=1, fill=WHITE, outline=BLUE_DARK, cell_w=250)
    rounded_box(draw, (560, 270, 1020, 700), "sym_buffer.map()", "Map a local workspace pointer into the remote rank's symmetric address space.\nWrite into src_token_topk_idx[local_expert][src_rank][dst_slot].", ORANGE, ORANGE_DARK)
    rounded_box(draw, (1110, 210, 1520, 760), "Destination rank queue", "Remote workspace now has a compact list of source token-topk slots for this local expert.", TEAL, TEAL_DARK)
    token_grid(draw, (1165, 390), ["slot0=T0,k0", "slot1=T1,k0", "slot2=T2,k1", "slot3=...", "slot4=...", "slot5=..."], cols=1, fill=WHITE, outline=TEAL_DARK, cell_w=270)
    arrow(draw, (470, 485), (560, 485), ORANGE_DARK, label="compute dst_rank, dst_slot")
    arrow(draw, (1020, 485), (1110, 485), TEAL_DARK, label="remote write over NVLink")
    footer(draw, "Refs: sm100_fp8_fp4_mega_moe.cuh:397-404, layout/mega_moe.cuh:152-160, layout/sym_buffer.cuh:33-37")
    return save(img, "dispatch_step_3_write_source_indices.png")


def step4() -> Path:
    img, draw = new_canvas()
    banner(draw, "STEP 4", "Finalize counts and cross-rank barrier", "Before any pull starts, the kernel publishes final receive counts and forces all ranks to observe a consistent routing state.")
    rounded_box(draw, (70, 230, 460, 730), "All SMs", "Grid sync ensures every CTA finished writing source index queues before SM0 publishes final expert counts.", BLUE, BLUE_DARK)
    token_grid(draw, (125, 400), ["SM0", "SM1", "SM2", "SM3"], cols=2, fill=WHITE, outline=BLUE_DARK, cell_w=110)
    rounded_box(draw, (545, 230, 1015, 730), "SM0 publishes recv counts", "For each expert:\nwrite expert_recv_count[src_rank][local_expert]\nadd into expert_recv_count_sum[local_expert].", ORANGE, ORANGE_DARK)
    rounded_box(draw, (1095, 230, 1525, 730), "NVLink barrier", "Only after every rank reaches the same barrier may dispatch warps start pulling token payloads.", RED, RED_DARK)
    arrow(draw, (460, 480), (545, 480), ORANGE_DARK, label="grid_sync")
    arrow(draw, (1015, 480), (1095, 480), RED_DARK, label="nvlink_barrier")
    pill(draw, (1118, 560, 1500, 625), "Guarantee: counts and src queues are globally visible", fill="#FFEAEA", outline=RED_DARK)
    footer(draw, "Refs: sm100_fp8_fp4_mega_moe.cuh:406-436, comm/barrier.cuh:9-71")
    return save(img, "dispatch_step_4_publish_counts_and_barrier.png")


def step5() -> Path:
    img, draw = new_canvas()
    banner(draw, "STEP 5", "Choose which rank supplies each pool slot", "The destination rank reconstructs the merged expert pool by reading per-rank counts and doing iterative min-peeling.")
    rounded_box(draw, (60, 215, 460, 760), "Cached recv counts", "scheduler.fetch_expert_recv_count() waits until expert_recv_count_sum is finalized, then caches token counts per local expert.", BLUE, BLUE_DARK)
    token_grid(draw, (120, 405), ["R0=3", "R1=1", "R2=2"], cols=3, fill=WHITE, outline=BLUE_DARK, cell_w=90)
    rounded_box(draw, (525, 215, 1040, 760), "Round-robin min-peeling", "Round 1: active ranks R0,R1,R2, length=1\nemit: R0 R1 R2\nRound 2: remaining R0=2,R2=1, length=1\nemit: R0 R2\nRound 3: remaining R0=1\nemit: R0", ORANGE, ORANGE_DARK)
    token_grid(draw, (635, 520), ["slot0", "R0", "slot1", "R1", "slot2", "R2", "slot3", "R0", "slot4", "R2", "slot5", "R0"], cols=3, fill=WHITE, outline=ORANGE_DARK, cell_w=92)
    rounded_box(draw, (1120, 215, 1530, 760), "Resolved source location", "For one pool slot, dispatch warp derives:\nsource rank\nsource token index inside that rank\nsource top-k slot", TEAL, TEAL_DARK)
    token_grid(draw, (1185, 465), ["rank=R2", "token=T17", "topk=k1"], cols=1, fill=WHITE, outline=TEAL_DARK, cell_w=180)
    arrow(draw, (460, 490), (525, 490), ORANGE_DARK, label="load per-rank counts")
    arrow(draw, (1040, 490), (1120, 490), TEAL_DARK, label="map slot -> source")
    footer(draw, "Refs: sm100_fp8_fp4_mega_moe.cuh:446-541")
    return save(img, "dispatch_step_5_round_robin_pull_plan.png")


def step6() -> Path:
    img, draw = new_canvas()
    banner(draw, "STEP 6", "Pull token payload into the local L1 pool", "Once the source location is known, the warp fetches token, scale factor, and top-k weight, then signals L1 readiness.")
    rounded_box(draw, (50, 210, 430, 770), "Remote source rank", "Inputs live on the source rank:\nFP8 token\nSF tensor\ntop-k weight", BLUE, BLUE_DARK)
    token_grid(draw, (110, 420), ["token[T17]", "sf[T17]", "weight[T17,k1]"], cols=1, fill=WHITE, outline=BLUE_DARK, cell_w=200)
    rounded_box(draw, (500, 180, 900, 790), "Dispatch warp staging", "1) elected lane issues TMA load of token into per-warp shared buffer\n2) all lanes copy SF in parallel\n3) elected lane reads weight", ORANGE, ORANGE_DARK)
    token_grid(draw, (580, 520), ["shared pull buffer", "sf copy loop", "weight register"], cols=1, fill=WHITE, outline=ORANGE_DARK, cell_w=240)
    rounded_box(draw, (980, 150, 1540, 820), "Local L1 expert pool", "After TMA load completes:\nstore token into l1_token_buffer\nstore SF into l1_sf_buffer\nstore weight into l1_topk_weights_buffer\nwrite TokenSrcMetadata\nred_add l1_arrival_count", TEAL, TEAL_DARK)
    token_grid(draw, (1070, 500), ["l1_token_buffer[pool]", "l1_sf_buffer[pool]", "l1_topk_weights[pool]", "TokenSrcMetadata", "l1_arrival_count++"], cols=1, fill=WHITE, outline=TEAL_DARK, cell_w=300)
    arrow(draw, (430, 480), (500, 480), ORANGE_DARK, label="TMA load + SF copy")
    arrow(draw, (900, 480), (980, 480), TEAL_DARK, label="TMA store + metadata + arrival")
    footer(draw, "Refs: sm100_fp8_fp4_mega_moe.cuh:544-599")
    return save(img, "dispatch_step_6_pull_token_sf_weight.png")


def step7() -> Path:
    img, draw = new_canvas()
    banner(draw, "STEP 7", "Cleanup workspace for the next kernel use", "This runs after dispatch work is done and is overlapped with combine reduction on the epilogue side.")
    rounded_box(draw, (70, 230, 440, 740), "SM0 cleanup", "Clear expert_send_count for all experts. This resets the global count-and-finished-status array.", BLUE, BLUE_DARK)
    rounded_box(draw, (520, 230, 1020, 740), "Other SMs cleanup", "For local experts assigned to this SM:\nclear expert_recv_count_sum\nclear per-rank expert_recv_count\nclear l1_arrival_count and l2_arrival_mask", ORANGE, ORANGE_DARK)
    rounded_box(draw, (1100, 230, 1520, 740), "Final NVLink barrier", "All ranks wait until workspace cleanup is done, so the next launch sees a clean routing state.", RED, RED_DARK)
    arrow(draw, (440, 485), (520, 485), ORANGE_DARK, label="parallel cleanup")
    arrow(draw, (1020, 485), (1100, 485), RED_DARK, label="barrier after cleanup")
    footer(draw, "Refs: sm100_fp8_fp4_mega_moe.cuh:602-649")
    return save(img, "dispatch_step_7_cleanup.png")


def main() -> None:
    paths = [step1(), step2(), step3(), step4(), step5(), step6(), step7()]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
