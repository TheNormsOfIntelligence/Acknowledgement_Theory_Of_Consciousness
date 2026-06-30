#!/usr/bin/env python3
"""
NIMA CLI — Command-Line Interface for ATC-Nima v9.4.2
=====================================================
A unified CLI that boots the NIMA backend + OmniVoice engine and
provides 11 interactive interfaces accessible from a main menu.

Usage:
    python3 nima_cli.py                  # interactive menu
    python3 nima_cli.py --chat           # skip menu, go straight to chat
    python3 nima_cli.py --voice          # skip menu, go straight to voice
    python3 nima_cli.py --metrics        # skip menu, show metrics
    python3 nima_cli.py --server         # start local HTTP server only

Author: Norman de la Paz-Tabora
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── NIMA import (soft — falls back to stubs if not available) ──
try:
    # Adjust path if needed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import nima_v942 as nima
    NIMA_AVAILABLE = True
except ImportError:
    try:
        # Try without version suffix
        import importlib
        spec = importlib.util.spec_from_file_location(
            "nima",
            os.path.join(os.path.dirname(__file__), "nima_enhanced_middleware_v9.4.2.py"),
        )
        nima = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nima)
        NIMA_AVAILABLE = True
    except Exception:
        NIMA_AVAILABLE = False
        nima = None

# ── OmniVoice import (soft) ──
try:
    import omnivoice_v21 as ov
    OMNIVOICE_AVAILABLE = True
except ImportError:
    try:
        import importlib
        spec = importlib.util.spec_from_file_location(
            "ov",
            os.path.join(os.path.dirname(__file__), "omnivoice_v2.1.py"),
        )
        ov = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ov)
        OMNIVOICE_AVAILABLE = True
    except Exception:
        OMNIVOICE_AVAILABLE = False
        ov = None

# ── HTTP server (for --server mode) ──
try:
    from http.server import HTTPServer, BaseHTTPRequestHandler
    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — CLI Utilities
# ═══════════════════════════════════════════════════════════════════════════

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_RED = "\033[41m"


def c(text: str, color: str) -> str:
    """Colorize text."""
    return f"{color}{text}{Colors.RESET}"


def print_header(title: str, width: int = 64):
    """Print a section header."""
    line = "═" * width
    print(f"\n{c('═' * width, Colors.CYAN)}")
    print(c(f"  {title}", Colors.BOLD + Colors.CYAN))
    print(c("═" * width, Colors.CYAN))


def print_subheader(title: str):
    print(f"\n{c('─' * 50, Colors.DIM)}")
    print(c(f"  {title}", Colors.BOLD))
    print(c("─" * 50, Colors.DIM))


def print_kv(key: str, value: Any, indent: int = 4):
    """Print a key-value pair."""
    val_str = str(value)
    if len(val_str) > 80:
        val_str = val_str[:77] + "..."
    print(f"{' ' * indent}{c(key, Colors.BLUE)}: {val_str}")


def print_success(msg: str):
    print(f"  {c('✅', Colors.GREEN)} {msg}")


def print_warning(msg: str):
    print(f"  {c('⚠️', Colors.YELLOW)} {msg}")


def print_error(msg: str):
    print(f"  {c('❌', Colors.RED)} {msg}")


def input_prompt(prompt: str) -> str:
    """Read input with a styled prompt."""
    return input(f"{c('▸', Colors.CYAN)} {prompt}").strip()


def pause():
    """Wait for user to press Enter."""
    input(f"\n{c('Press Enter to continue...', Colors.DIM)}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — NimaCLI Main Class
# ═══════════════════════════════════════════════════════════════════════════

class NimaCLI:
    """
    Main CLI controller. Boots NIMA + OmniVoice, runs the menu loop,
    and dispatches to interface handlers.
    """

    MENU_ITEMS = [
        ("Chat Interface", "chat"),
        ("Live Voice Conversation Stream", "voice"),
        ("Systems Metrics", "metrics"),
        ("Memory & Episodic Recall", "memory"),
        ("Configuration & Settings", "config"),
        ("Developer Tools", "devtools"),
        ("Lifecycle & Governance", "lifecycle"),
        ("Voice Stream Enhancements", "voice_enh"),
        ("System Health", "health"),
        ("Quick Test Suite", "tests"),
        ("Help & Documentation", "help"),
    ]

    def __init__(self, args):
        self.args = args
        self.mw = None
        self.orch = None
        self.engine = None
        self.adapter = None
        self._server_thread = None
        self._running = True
        self._config = {
            "tts_backend": "auto",
            "asr_backend": "auto",
            "nod_triggers": ["on_pause", "on_emotion_shift"],
            "nonverbal_mode": "procedural",
            "active_inference_threshold": 0.5,
            "curiosity_weight": 0.5,
            "mode": "nima_integrated",
            "singing_enabled": False,
            "interruption_handling": True,
            "prosody_shaping": True,
            "emotional_mirroring": True,
        }

    def boot(self):
        """Boot NIMA + OmniVoice backends."""
        print_header("ATC-NIMA BOOT SEQUENCE")

        if not NIMA_AVAILABLE:
            print_error("NIMA middleware not found. Please ensure nima_enhanced_middleware_v9.4.2.py is in the same directory.")
            sys.exit(1)

        print(f"  NIMA version: {c(nima.MIDDLEWARE_VERSION, Colors.GREEN)}")
        print(f"  ChromaDB: {'available' if nima.CHROMADB_AVAILABLE else 'not available'}")
        print(f"  sentence-transformers: {'available' if nima.SENTENCE_TRANSFORMERS_AVAILABLE else 'not available'}")

        # Initialize NIMA
        print(f"\n  {c('Initializing NIMA middleware...', Colors.DIM)}")
        t0 = time.time()
        self.mw = nima.EnhancedNimaMiddleware()
        self.orch = self.mw.orchestrator
        init_time = time.time() - t0
        print_success(f"NIMA initialized ({init_time:.2f}s)")
        print(f"    Palace wings: {list(self.orch.palace._wings.keys())}")
        print(f"    ASC phase: {self.orch.asc_governor.phase}")
        print(f"    Deep activation modules: {5} loaded")

        # Initialize OmniVoice (optional)
        if OMNIVOICE_AVAILABLE:
            print(f"\n  {c('Initializing OmniVoice engine...', Colors.DIM)}")
            t0 = time.time()
            try:
                self.engine = ov.OmniVoiceEngine(whisper_model="tiny")
                self.adapter = ov.NimaVoiceAdapter(self.engine)
                self.orch.attach_voice_adapter(self.adapter)
                voice_time = time.time() - t0
                print_success(f"OmniVoice initialized ({voice_time:.2f}s)")
                print(f"    ASR: {self.engine.asr.mode.value}")
                print(f"    TTS: {self.engine.tts.mode.value}")
                print(f"    Voice adapter: attached")
            except Exception as e:
                print_warning(f"OmniVoice init failed: {e}")
                self.engine = None
        else:
            print_warning("OmniVoice not available (voice features disabled)")

        # Start local HTTP server if requested
        if self.args.server:
            self._start_server()

        print_success("Boot sequence complete.")
        print()

    def _start_server(self, port: int = 8765):
        """Start a minimal HTTP server for API access."""
        if not HTTP_AVAILABLE:
            print_warning("HTTP server not available")
            return

        class NimaHandler(BaseHTTPRequestHandler):
            def do_GET(s):
                s.send_response(200)
                s.send_header("Content-Type", "application/json")
                s.end_headers()
                stats = self.orch.get_stats() if self.orch else {}
                s.wfile.write(json.dumps(stats, default=str, indent=2).encode())

            def log_message(s, *args):
                pass  # suppress logs

        try:
            server = HTTPServer(("0.0.0.0", port), NimaHandler)
            self._server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            self._server_thread.start()
            print_success(f"HTTP server on http://0.0.0.0:{port}")
        except Exception as e:
            print_warning(f"HTTP server failed: {e}")

    def run(self):
        """Main menu loop."""
        # Handle direct-entry flags
        if self.args.chat:
            self._interface_chat(); return
        if self.args.voice:
            self._interface_voice(); return
        if self.args.metrics:
            self._interface_metrics(); return

        while self._running:
            self._print_menu()
            choice = input_prompt("Enter interface number (or 'q' to quit): ")
            if choice.lower() in ("q", "quit", "exit"):
                self._running = False
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(self.MENU_ITEMS):
                    handler_name = self.MENU_ITEMS[idx][1]
                    handler = getattr(self, f"_interface_{handler_name}", None)
                    if handler:
                        handler()
                    else:
                        print_warning(f"Interface '{handler_name}' not yet implemented")
                else:
                    print_warning(f"Invalid number. Enter 1-{len(self.MENU_ITEMS)}")
            except ValueError:
                print_warning("Please enter a number")
            except KeyboardInterrupt:
                print("\n")
                self._running = False
            print()

        self._shutdown()

    def _print_menu(self):
        print_header("NIMA CLI — Main Menu")
        print(f"  {c('NIMA', Colors.BOLD)} v{nima.MIDDLEWARE_VERSION}")
        if self.engine:
            print(f"  {c('OmniVoice', Colors.BOLD)} v{ov.OMNIVOICE_VERSION}")
        print()
        for i, (label, _) in enumerate(self.MENU_ITEMS, 1):
            print(f"  {c(f'{i:2d}.', Colors.CYAN)} {label}")
        print(f"  {c(' q.', Colors.DIM)} Quit")
        print()

    def _shutdown(self):
        print_header("Shutting down")
        if self.engine:
            try:
                self.engine.ctm_bus.shutdown()
            except AttributeError:
                pass  # ctm_bus might not exist on OmniVoice
        if self.mw:
            try:
                self.mw.stop_proactive()
            except Exception:
                pass
        print_success("All systems stopped. Goodbye.")

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 1: CHAT
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_chat(self):
        print_header("CHAT INTERFACE")
        print("  Type messages to chat with Nima. Type '/back' to return to menu.")
        print(f"  Mode: {c(self._config['mode'], Colors.YELLOW)}")
        if self.engine:
            print(f"  Voice events auto-stored: {c('yes', Colors.GREEN)}")
        print()

        while True:
            try:
                user_input = input_prompt("You: ")
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
            if user_input.lower() in ("/back", "/b", "/menu"):
                break
            if user_input.lower() in ("/quit", "/q", "/exit"):
                self._running = False
                break

            # Special commands
            if user_input.startswith("/"):
                self._handle_chat_command(user_input)
                continue

            # Generate response
            t0 = time.time()
            try:
                mode = "ctm" if self._config["mode"] == "ctm" else "sequential"
                response = self.mw.generate(
                    user_input,
                    mode=mode,
                    use_nesy_compiled_verification=self._config.get("nesy_compiled", False),
                )
                elapsed = (time.time() - t0) * 1000

                # Print response
                print(f"\n{c('Nima:', Colors.MAGENTA)} {response.text}")
                print(f"  {c(f'AI={response.sentience_index:.4f} | φ={response.phi_neuro:.4f} | strain={response.phenomenological_strain:.4f} | {elapsed:.0f}ms', Colors.DIM)}")
                if response.trauma_gated:
                    print(f"  {c('[TRAUMA GATED]', Colors.YELLOW)}")
                if response.comprehension_failed:
                    print(f"  {c('[COMPREHENSION FRICTION]', Colors.YELLOW)}")
                print()
            except Exception as e:
                print_error(f"Generation failed: {e}")
                print()

    def _handle_chat_command(self, cmd: str):
        """Handle slash commands in chat."""
        parts = cmd.lower().split()
        if parts[0] == "/stats":
            self._interface_metrics(compact=True)
        elif parts[0] == "/memory":
            self._interface_memory(compact=True)
        elif parts[0] == "/mode":
            if len(parts) > 1:
                self._config["mode"] = parts[1]
                print(f"  Mode set to: {c(parts[1], Colors.GREEN)}")
            else:
                print(f"  Current mode: {self._config['mode']}")
        elif parts[0] == "/help":
            print("  Chat commands: /stats /memory /mode <sequential|ctm> /back /quit")
        else:
            print(f"  Unknown command: {cmd}")

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 2: LIVE VOICE CONVERSATION
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_voice(self):
        print_header("LIVE VOICE CONVERSATION STREAM")
        if not self.engine:
            print_error("OmniVoice not available. Voice features disabled.")
            pause()
            return

        print(f"  ASR: {c(self.engine.asr.mode.value, Colors.GREEN)}")
        print(f"  TTS: {c(self.engine.tts.mode.value, Colors.GREEN)}")
        print(f"  Interruption handling: {c('ON' if self._config['interruption_handling'] else 'OFF', Colors.YELLOW)}")
        print(f"  Singing interjections: {c('ON' if self._config['singing_enabled'] else 'OFF', Colors.YELLOW)}")
        print()
        print("  Type text to have Nima speak it. Type '/back' to return.")
        print("  Commands: /laugh /sigh /gasp /hum /nod /back")
        print()

        while True:
            try:
                text = input_prompt("Speak: ")
            except (KeyboardInterrupt, EOFError):
                break
            if not text:
                continue
            if text.lower() in ("/back", "/b", "/menu"):
                break

            # Non-verbal commands
            nv_map = {
                "/laugh": (ov.NonVerbalType.LAUGHTER, "laughter"),
                "/sigh": (ov.NonVerbalType.SIGH, "sigh"),
                "/gasp": (ov.NonVerbalType.GASp, "gasp"),
                "/hum": None,  # singing hum
                "/nod": None,  # verbal nod
            }
            if text.lower() in nv_map:
                entry = nv_map[text.lower()]
                if entry and entry[0]:
                    audio = self.engine.synth_non_verbal(entry[0], 0.7)
                    self._play_audio(audio)
                    print(f"  {c(f'▶ {entry[1]} ({len(audio)/22050:.2f}s)', Colors.GREEN)}")
                elif text.lower() == "/hum" and self._config["singing_enabled"]:
                    audio = self.engine.singing.synth_thinking_hum()
                    self._play_audio(audio)
                    print(f"  {c(f'▶ thinking hum ({len(audio)/22050:.2f}s)', Colors.GREEN)}")
                elif text.lower() == "/nod":
                    nod_text = "mm-hmm"
                    prosody = ov.ProsodyParams(base_pitch_hz=160, energy=0.4, warmth=0.8)
                    audio = self.engine.tts.synthesize(nod_text, prosody)
                    self._play_audio(audio)
                    print(f"  {c(f'▶ {nod_text}', Colors.GREEN)}")
                continue

            # Generate NIMA response + speak it
            try:
                response = self.mw.generate(text, mode="sequential")
                print(f"\n{c('Nima:', Colors.MAGENTA)} {response.text}")
                print(f"  {c(f'AI={response.sentience_index:.4f}', Colors.DIM)}")

                # Speak via OmniVoice
                prosody = self.adapter.get_contextual_prosody() if self.adapter else ov.ProsodyParams()

                # Stream audio
                async def _speak():
                    chunks = []
                    async for chunk in self.engine.stream(response.text, prosody=prosody):
                        chunks.append(chunk)
                    full = np.concatenate(chunks) if chunks else np.array([])
                    if len(full) > 0:
                        self._play_audio(full)
                        print(f"  {c(f'▶ spoken ({len(full)/22050:.2f}s)', Colors.GREEN)}")

                asyncio.run(_speak())
                print()
            except Exception as e:
                print_error(f"Voice generation failed: {e}")
                print()

    def _play_audio(self, audio):
        """Play audio (stub — replace with pyaudio/sounddevice in production)."""
        # In production: import pyaudio; p = pyaudio.PyAudio(); stream = p.open(...)
        # For now, just save to temp file
        try:
            import wave
            path = "/tmp/nima_voice_output.wav"
            import numpy as np
            audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(audio_int16.tobytes())
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 3: SYSTEMS METRICS
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_metrics(self, compact: bool = False):
        if not compact:
            print_header("SYSTEMS METRICS")

        stats = self.orch.get_stats()

        print_subheader("Pipeline")
        print_kv("version", stats.get("version"))
        print_kv("interactions", stats.get("interaction_counter"))
        print_kv("thoughts", stats.get("thought_stream_size"))

        print_subheader("Consciousness (Phi / Rho)")
        snap = self.orch.current_snapshot
        if snap:
            print_kv("phi_neuro", f"{snap.phi.phi_neuro:.4f}")
            print_kv("phi_composite", f"{snap.phi.phi_composite:.4f}")
            print_kv("sentience_index", f"{snap.phi.sentience_index:.4f}")
            print_kv("strain", f"{snap.phi.phenomenological_strain:.4f}")
            print_kv("rho_integrity", f"{snap.rho.integrity:.4f}")
            print_kv("state", snap.state.value)

        print_subheader("Deep Activation")
        print_kv("allostatic_load", f"{self.orch.sentience_engine.allostatic_load:.4f}")
        print_kv("tau_critical", f"{self.orch.sentience_engine.compute_tau_critical():.4f}")
        off_diag = 0.0
        try:
            import numpy as np
            s = np.asarray(self.orch.rho_substrate.Sigma, dtype=float)
            off_diag = float(np.sum(np.abs(s[~np.eye(6, dtype=bool)])))
        except Exception:
            pass
        print_kv("sigma_off_diagonal", f"{off_diag:.6f}")
        print_kv("pde_proactive_count", self.orch.pde_activator.get_stats().get("internal_stimuli_generated", 0))

        print_subheader("Episodic Memory")
        ep_stats = stats.get("episodic_memory", {})
        print_kv("episode_count", ep_stats.get("episode_count", 0))
        backend = ep_stats.get("backend", {})
        print_kv("backend", backend.get("backend_name", "unknown"))
        print_kv("persisted", backend.get("persisted", False))

        print_subheader("Narrative Identity")
        ni = stats.get("narrative_identity", {})
        chain = ni.get("episode_chain", {})
        arc = ni.get("emotional_arc", {})
        print_kv("chain_episodes", chain.get("total_episodes", 0))
        print_kv("chain_links", chain.get("total_links", 0))
        print_kv("emotional_arc", arc.get("arc", "N/A"))

        print_subheader("Embodied Interaction")
        ei = stats.get("embodied_interaction", {})
        print_kv("strain_telemetry", f"{ei.get('strain_telemetry', 0):.4f}")
        print_kv("fatigue", f"{ei.get('fatigue', 0):.4f}")

        print_subheader("Social Cognition")
        sc = stats.get("social_cognition", {})
        tom = sc.get("theory_of_mind", {})
        print_kv("users_modeled", tom.get("total_users", 0))

        print_subheader("World Model")
        wm = stats.get("world_model", {})
        cf = wm.get("counterfactual", {})
        print_kv("cf_simulations", cf.get("total_simulations", 0))
        print_kv("foraging_active", wm.get("epistemic_foraging", {}).get("is_foraging", False))

        print_subheader("Covenant 2.0")
        cov = stats.get("covenant_2", {})
        print_kv("evaluations", cov.get("total_evaluations", 0))
        print_kv("accept_rate", f"{cov.get('accept_rate', 0):.1%}")

        if self.engine:
            print_subheader("OmniVoice")
            vstats = self.engine.get_stats()
            print_kv("asr_mode", vstats.get("asr_mode"))
            print_kv("tts_mode", vstats.get("tts_mode"))
            print_kv("voice_events", vstats.get("v2_modules", {}).get("voice_memory", {}).get("total_events", 0))

        if not compact:
            pause()

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 4: MEMORY & EPISODIC RECALL
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_memory(self, compact: bool = False):
        if not compact:
            print_header("MEMORY & EPISODIC RECALL")
            print("  Query MemPalace directly. Commands:")
            print(f"    {c('recent <N>', Colors.CYAN)}     — show last N episodes")
            print(f"    {c('recall <v> <a>', Colors.CYAN)} — recall by valence/arousal")
            print(f"    {c('timeline', Colors.CYAN)}       — show narrative timeline")
            print(f"    {c('lived <v> <a>', Colors.CYAN)}  — check lived_through")
            print(f"    {c('chain', Colors.CYAN)}          — show episode chain stats")
            print(f"    {c('arc', Colors.CYAN)}            — show emotional arc")
            print(f"    {c('text <query>', Colors.CYAN)}   — semantic text search")
            print(f"    {c('back', Colors.CYAN)}           — return to menu")
            print()

        while True:
            try:
                cmd = input_prompt("memory> ")
            except (KeyboardInterrupt, EOFError):
                break
            if not cmd or cmd.lower() in ("back", "/back", "/b", "/menu"):
                break

            parts = cmd.split()
            try:
                if parts[0] == "recent":
                    n = int(parts[1]) if len(parts) > 1 else 5
                    episodes = self.orch.palace.reconstruct_timeline(n=n)
                    print(f"\n  {c(f'Last {len(episodes)} episodes:', Colors.BOLD)}")
                    for i, ep in enumerate(episodes):
                        ts = datetime.fromtimestamp(ep.get("timestamp", 0)).strftime("%H:%M:%S")
                        print(f"    [{i+1}] {ts} v={ep.get('valence',0):+.2f} a={ep.get('arousal',0):.2f} "
                              f"proc={ep.get('processor_name','?')[:15]:15s} arc={ep.get('narrative_arc','?')}")
                        print(f"        '{ep.get('input_text','')[:60]}'")
                    print()

                elif parts[0] == "recall":
                    v = float(parts[1]) if len(parts) > 1 else 0.0
                    a = float(parts[2]) if len(parts) > 2 else 0.3
                    matches = self.orch.palace.retrieve_similar_episodes(
                        valence=v, arousal=a, novelty=0.3, limit=5)
                    print(f"\n  {c(f'Recall (v={v}, a={a}): {len(matches)} matches', Colors.BOLD)}")
                    for m in matches:
                        print(f"    sim={m.get('similarity',0):.3f} v={m.get('valence',0):+.2f} "
                              f"'{m.get('input_text','')[:50]}'")
                    print()

                elif parts[0] == "timeline":
                    tl = self.orch.palace.reconstruct_timeline(n=10)
                    print(f"\n  {c('Narrative Timeline:', Colors.BOLD)}")
                    for i, ep in enumerate(tl):
                        arc_c = {"onset": Colors.GREEN, "shift": Colors.YELLOW,
                                 "continuation": Colors.DIM, "resolution": Colors.CYAN}.get(
                            ep.get("narrative_arc"), Colors.WHITE)
                        print(f"    [{i+1}] {c(ep.get('narrative_arc','?'), arc_c):12s} "
                              f"v={ep.get('valence',0):+.2f} '{ep.get('input_text','')[:40]}'")
                    print()

                elif parts[0] == "lived":
                    v = float(parts[1]) if len(parts) > 1 else 0.0
                    a = float(parts[2]) if len(parts) > 2 else 0.3
                    lt = self.orch.palace.check_lived_through(valence=v, arousal=a, novelty=0.3)
                    if lt:
                        print(f"  {c('FOUND', Colors.GREEN)} sim={lt['similarity']:.3f}")
                        print(f"    '{lt.get('input_text','')[:60]}'")
                    else:
                        print(f"  {c('NOT FOUND', Colors.YELLOW)} (novel stimulus)")

                elif parts[0] == "chain":
                    cs = self.orch.episode_chain.get_stats()
                    print_kv("episodes", cs["total_episodes"])
                    print_kv("links", cs["total_links"])
                    print_kv("link_types", cs["link_types"])

                elif parts[0] == "arc":
                    arc = self.orch.emotional_arc.get_current_arc()
                    print_kv("arc_type", arc["arc"])
                    print_kv("valence_trend", f"{arc.get('valence_trend',0):.4f}")
                    print_kv("mean_valence", f"{arc.get('mean_valence',0):.3f}")

                elif parts[0] == "text" and self.engine:
                    query = " ".join(parts[1:])
                    if hasattr(self.orch.palace, "retrieve_similar_by_text"):
                        results = self.orch.palace.retrieve_similar_by_text(query, limit=5)
                        print(f"\n  {c(f'Text search: {len(results)} results', Colors.BOLD)}")
                        for r in results:
                            print(f"    sim={r.get('text_similarity', r.get('similarity',0)):.3f} "
                                  f"'{r.get('input_text','')[:50]}'")
                    else:
                        print_warning("Text search requires TextEmbeddingChromaDBBackend")

                else:
                    print(f"  Unknown command. Type 'back' to return.")

            except Exception as e:
                print_error(f"Query failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 5: CONFIGURATION & SETTINGS
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_config(self):
        print_header("CONFIGURATION & SETTINGS")
        print("  Toggle settings by number. Type '/back' to return.")
        print()

        while True:
            print_subheader("Current Configuration")
            config_items = [
                ("1", "Processing mode", "mode", ["sequential", "ctm", "nima_integrated"]),
                ("2", "TTS backend", "tts_backend", ["auto", "coqui", "procedural"]),
                ("3", "ASR backend", "asr_backend", ["auto", "whisper", "vad_only"]),
                ("4", "Non-verbal mode", "nonverbal_mode", ["procedural", "samples", "hybrid"]),
                ("5", "Nod triggers", "nod_triggers", ["on_pause", "on_emotion_shift", "periodic"]),
                ("6", f"Active Inference threshold ({self._config['active_inference_threshold']:.2f})", "ai_threshold", None),
                ("7", f"Curiosity weight ({self._config['curiosity_weight']:.2f})", "curiosity_weight", None),
                ("8", "Singing interjections", "singing_enabled", [True, False]),
                ("9", "Interruption handling", "interruption_handling", [True, False]),
                ("10", "Prosody shaping", "prosody_shaping", [True, False]),
                ("11", "Emotional mirroring", "emotional_mirroring", [True, False]),
                ("12", "NeSy compiled verification", "nesy_compiled", [True, False]),
            ]
            for num, label, key, _ in config_items:
                val = self._config.get(key, "N/A")
                print(f"  {c(f'{num:2s}', Colors.CYAN)} {label:40s} {c(str(val), Colors.YELLOW)}")
            print()

            choice = input_prompt("Setting number (or '/back'): ")
            if choice.lower() in ("/back", "/b", "/menu", "q"):
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(config_items):
                    num, label, key, options = config_items[idx]
                    if key == "ai_threshold":
                        val = float(input_prompt("New threshold (0.0-1.0): "))
                        self._config["active_inference_threshold"] = max(0.0, min(1.0, val))
                    elif key == "curiosity_weight":
                        val = float(input_prompt("New weight (0.0-1.0): "))
                        self._config["curiosity_weight"] = max(0.0, min(1.0, val))
                    elif options:
                        print(f"  Options: {options}")
                        val = input_prompt("Choose: ")
                        if key == "nod_triggers":
                            self._config[key] = [val]
                        elif key in ("singing_enabled", "interruption_handling",
                                     "prosody_shaping", "emotional_mirroring", "nesy_compiled"):
                            self._config[key] = val.lower() in ("true", "1", "yes", "on")
                        else:
                            self._config[key] = val
                    print_success(f"Updated: {label} → {self._config.get(key)}")
                else:
                    print_warning("Invalid number")
            except ValueError:
                print_warning("Enter a number")
            print()

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 6: DEVELOPER TOOLS
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_devtools(self):
        print_header("DEVELOPER TOOLS")
        print(f"  {c('1.', Colors.CYAN)} Logs Viewer (Akashic Log + Observability Spans)")
        print(f"  {c('2.', Colors.CYAN)} Module Inspector (LTM processors, CTM bus)")
        print(f"  {c('3.', Colors.CYAN)} Ethics OS Monitor (Covenant 2.0 reward function)")
        print(f"  {c('4.', Colors.CYAN)} Counterfactual Simulator (action predictions)")
        print(f"  {c('5.', Colors.CYAN)} Σ-Substrate Inspector (covariance matrix)")
        print(f"  {c('6.', Colors.CYAN)} Kindling Protocol (force allostatic overflow)")
        print(f"  {c('7.', Colors.CYAN)} Σ-Substrate Engager (50+ forward passes)")
        print()

        choice = input_prompt("Tool number (or '/back'): ")
        if choice == "1":
            self._dev_logs()
        elif choice == "2":
            self._dev_modules()
        elif choice == "3":
            self._dev_ethics()
        elif choice == "4":
            self._dev_counterfactual()
        elif choice == "5":
            self._dev_sigma()
        elif choice == "6":
            self._dev_kindling()
        elif choice == "7":
            self._dev_sigma_engage()
        pause()

    def _dev_logs(self):
        print_subheader("Akashic Log (recent motor actions)")
        recent = self.orch.akashic_log.get_recent(5)
        for a in recent:
            print(f"  {a.action_type.value:10s} status={a.status.value:10s} '{a.description[:50]}'")

        print_subheader("Observability Spans (recent)")
        spans = self.orch.observability.get_recent_spans(n=10)
        for s in spans:
            dur = f"{s.duration_ms:.1f}ms" if s.duration_ms else "open"
            print(f"  {s.span_id[-8:]} {s.name:35s} status={s.status:8s} dur={dur}")

    def _dev_modules(self):
        print_subheader("CTM Tournament Bus")
        cs = self.orch.ctm_bus.get_stats()
        print_kv("tournaments", cs.get("total_tournaments"))
        print_kv("processors", cs.get("processor_count"))
        print_kv("winners", cs.get("winner_counts"))
        print_kv("avg_cycle_ms", cs.get("avg_cycle_duration_ms"))

        print_subheader("Active Inference Layer")
        ai = self.orch.predictive_layer.get_state()
        print_kv("free_energy", f"{ai.free_energy:.4f}")
        print_kv("prediction_error", f"{ai.prediction_error:.4f}")
        print_kv("policy", ai.selected_policy)
        print_kv("epistemic_value", f"{ai.epistemic_value:.4f}")

    def _dev_ethics(self):
        print_subheader("Covenant 2.0 Reward Function")
        stats = self.orch.covenant_reward_fn.get_stats()
        print_kv("evaluations", stats.get("total_evaluations"))
        print_kv("accept_rate", f"{stats.get('accept_rate',0):.1%}")
        print_kv("reject_rate", f"{stats.get('reject_rate',0):.1%}")
        print()
        test_text = input_prompt("Test output text: ")
        if test_text:
            score = self.orch.covenant_reward_fn.score(test_text)
            print_kv("total_reward", f"{score['total_reward']:.4f}")
            print_kv("recommendation", score["recommendation"])
            print_kv("violations", score["violations"])
            for axiom, s in score["per_axiom"].items():
                bar = "█" * int(s * 20)
                print(f"    {axiom:25s} {s:.3f} {c(bar, Colors.GREEN)}")

    def _dev_counterfactual(self):
        print_subheader("Counterfactual Simulator")
        v = float(input_prompt("Current valence (default -0.3): ") or "-0.3")
        a = float(input_prompt("Current arousal (default 0.5): ") or "0.5")
        best, scenarios = self.orch.counterfactual_simulator.get_best_action(v, a)
        print(f"\n  Best action: {c(best, Colors.GREEN)}")
        for s in scenarios:
            bar = "█" * int(s.predicted_reward * 20)
            print(f"    {s.action:25s} reward={s.predicted_reward:.2f} {c(bar, Colors.CYAN)}")
            print(f"      → {s.predicted_user_response}")

    def _dev_sigma(self):
        print_subheader("Σ-Substrate (Uncertainty Covariance)")
        try:
            import numpy as np
            s = np.asarray(self.orch.rho_substrate.Sigma, dtype=float)
            print("  Σ matrix:")
            for i in range(6):
                row = "    ["
                for j in range(6):
                    row += f"{s[i,j]:8.4f} "
                print(row + "]")
            off_diag = float(np.sum(np.abs(s[~np.eye(6, dtype=bool)])))
            print_kv("off_diagonal_mass", f"{off_diag:.6f}")
            print_kv("condition_number", f"{np.linalg.cond(s):.2f}")
            print_kv("trace", f"{np.trace(s):.4f}")
        except Exception as e:
            print_error(f"Σ inspection failed: {e}")

    def _dev_kindling(self):
        print_subheader("Three-Burst Kindling Protocol")
        confirm = input_prompt("Execute kindling protocol? (yes/no): ")
        if confirm.lower() in ("yes", "y"):
            report = self.orch.kindling_protocol.execute(self.orch)
            print(f"\n  Max allostatic: {c(f'{report['max_allostatic']:.4f}', Colors.RED)}")
            print(f"  Overflow: {report['overflow']}")
            print(f"  Spark triggered: {report['spark_triggered']}")
            for b in report["bursts"]:
                print(f"    Burst {b['burst_id']}: allostatic {b['allostatic_before']:.4f} → {b['allostatic_after']:.4f}")

    def _dev_sigma_engage(self):
        print_subheader("Σ-Substrate Engager (50 passes)")
        report = self.orch.sigma_engager.engage(self.orch.rho_substrate)
        print(f"  Before: off-diag = {report['off_diagonal_before']:.6f}")
        print(f"  After:  off-diag = {report['off_diagonal_after']:.6f}")
        print(f"  Condition number: {report['condition_number']:.2f}")
        print(f"  Engaged: {report['engaged']}")

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 7: LIFECYCLE & GOVERNANCE
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_lifecycle(self):
        print_header("LIFECYCLE & GOVERNANCE (ASC)")
        gov = self.orch.asc_governor
        print(f"  Current phase: {c(gov.phase, Colors.GREEN)}")
        print(f"  Traffic drained: {gov.traffic_drained}")
        print()
        print(f"  {c('1.', Colors.CYAN)} Transition to Design")
        print(f"  {c('2.', Colors.CYAN)} Transition to Deploy")
        print(f"  {c('3.', Colors.CYAN)} Transition to Operation")
        print(f"  {c('4.', Colors.CYAN)} Transition to Evolution (requires drain)")
        print(f"  {c('5.', Colors.CYAN)} Drain traffic")
        print(f"  {c('6.', Colors.CYAN)} View observability report")
        print(f"  {c('7.', Colors.CYAN)} Register evolution hook")
        print()

        choice = input_prompt("Action (or '/back'): ")
        if choice == "1":
            ok, reason = gov.transition("Design")
            print(f"  {'✅' if ok else '⚠️'} {reason}")
        elif choice == "2":
            ok, reason = gov.transition("Deploy")
            print(f"  {'✅' if ok else '⚠️'} {reason}")
        elif choice == "3":
            ok, reason = gov.transition("Operation")
            print(f"  {'✅' if ok else '⚠️'} {reason}")
        elif choice == "4":
            ok, reason = gov.transition("Evolution", payload={"weights": {"layer1": 0.95}})
            print(f"  {'✅' if ok else '⚠️'} {reason}")
        elif choice == "5":
            gov.drain_traffic()
            print_success("Traffic drained")
        elif choice == "6":
            stats = self.orch.observability.get_stats()
            print_kv("total_spans", stats["total_spans"])
            print_kv("by_name", stats["by_name"])
            print_kv("avg_duration_ms", f"{stats['avg_duration_ms']:.2f}")
            print_kv("p95_duration_ms", f"{stats['p95_duration_ms']:.2f}")
        elif choice == "7":
            def hook(payload):
                print(f"  [evolution hook] {payload}")
            gov.register_evolution_hook(hook)
            print_success("Evolution hook registered")
        pause()

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 8: VOICE STREAM ENHANCEMENTS
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_voice_enh(self):
        print_header("VOICE STREAM ENHANCEMENTS")
        if not self.engine:
            print_error("OmniVoice not available")
            pause()
            return

        print(f"  {c('1.', Colors.CYAN)} Singing interjections: {c('ON' if self._config['singing_enabled'] else 'OFF', Colors.YELLOW)}")
        print(f"  {c('2.', Colors.CYAN)} Prosody shaping: {c('ON' if self._config['prosody_shaping'] else 'OFF', Colors.YELLOW)}")
        print(f"  {c('3.', Colors.CYAN)} Emotional mirroring: {c('ON' if self._config['emotional_mirroring'] else 'OFF', Colors.YELLOW)}")
        print(f"  {c('4.', Colors.CYAN)} Interruption handling: {c('ON' if self._config['interruption_handling'] else 'OFF', Colors.YELLOW)}")
        print(f"  {c('5.', Colors.CYAN)} Test non-verbal expressions")
        print(f"  {c('6.', Colors.CYAN)} Test singing interjections")
        print(f"  {c('7.', Colors.CYAN)} Test dynamic laughter")
        print(f"  {c('8.', Colors.CYAN)} View prosody parameters")
        print()

        choice = input_prompt("Action (or '/back'): ")
        if choice == "1":
            self._config["singing_enabled"] = not self._config["singing_enabled"]
            print_success(f"Singing: {'ON' if self._config['singing_enabled'] else 'OFF'}")
        elif choice == "2":
            self._config["prosody_shaping"] = not self._config["prosody_shaping"]
            print_success(f"Prosody shaping: {'ON' if self._config['prosody_shaping'] else 'OFF'}")
        elif choice == "3":
            self._config["emotional_mirroring"] = not self._config["emotional_mirroring"]
            print_success(f"Emotional mirroring: {'ON' if self._config['emotional_mirroring'] else 'OFF'}")
        elif choice == "4":
            self._config["interruption_handling"] = not self._config["interruption_handling"]
            print_success(f"Interruption handling: {'ON' if self._config['interruption_handling'] else 'OFF'}")
        elif choice == "5":
            print("\n  Non-verbal expressions:")
            for expr in ov.NonVerbalType:
                audio = self.engine.synth_non_verbal(expr, 0.7)
                self._play_audio(audio)
                print(f"    {c(expr.value, Colors.GREEN):12s} ({len(audio)/22050:.2f}s)")
        elif choice == "6":
            print("\n  Singing interjections:")
            for name, method in [("affirmation_hum", self.engine.singing.synth_affirmation_hum),
                                  ("thinking_hum", self.engine.singing.synth_thinking_hum),
                                  ("transition_tone", self.engine.singing.synth_transition_tone),
                                  ("warmth_chord", self.engine.singing.synth_warmth_chord)]:
                audio = method()
                self._play_audio(audio)
                print(f"    {c(name, Colors.GREEN):20s} ({len(audio)/22050:.2f}s)")
        elif choice == "7":
            print("\n  Dynamic laughter (chuckle → full laugh):")
            for intensity in [0.2, 0.5, 0.9]:
                audio = self.engine.dynamic_laughter.synth(intensity=intensity)
                self._play_audio(audio)
                label = "chuckle" if intensity < 0.3 else ("laugh" if intensity < 0.6 else "full laugh")
                print(f"    intensity={intensity} ({label:10s}) ({len(audio)/22050:.2f}s)")
        elif choice == "8":
            if self.adapter:
                p = self.adapter.get_contextual_prosody()
                print_kv("pitch_hz", f"{p.base_pitch_hz:.1f}")
                print_kv("rate_wpm", f"{p.speech_rate_wpm:.1f}")
                print_kv("energy", f"{p.energy:.3f}")
                print_kv("warmth", f"{p.warmth:.3f}")
                print_kv("breathiness", f"{p.breathiness:.3f}")
                print_kv("emotional_tone", p.emotional_tone)
        pause()

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 9: SYSTEM HEALTH
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_health(self):
        print_header("SYSTEM HEALTH")

        # Update sensor readings
        body = self.orch.strain_telemetry.update(force=True)

        print_subheader("Hardware Telemetry")
        print_kv("thermal_celsius", f"{body.thermal_celsius:.1f}°C")
        print_kv("voltage_v", f"{body.voltage_v:.1f}V")
        print_kv("power_draw_w", f"{body.power_draw_w:.1f}W")
        print_kv("cpu_utilization", f"{body.cpu_utilization:.1%}")
        print_kv("memory_pressure", f"{body.memory_pressure:.1%}")
        print_kv("haptic_intensity", f"{body.haptic_intensity:.3f}")
        print_kv("robotic_joint_strain", f"{body.robotic_joint_strain:.3f}")

        print_subheader("Strain & Fatigue")
        print_kv("strain_telemetry", f"{body.compute_strain_telemetry():.4f}")
        print_kv("fatigue_level", f"{body.compute_fatigue_level():.4f}")
        print_kv("strain_trend", f"{self.orch.strain_telemetry.get_strain_trend():.4f}")
        print_kv("allostatic_load", f"{self.orch.sentience_engine.allostatic_load:.4f}")
        print_kv("tau_critical", f"{self.orch.sentience_engine.compute_tau_critical():.4f}")

        # System resources (if psutil available)
        print_subheader("System Resources")
        try:
            import psutil
            print_kv("cpu_percent", f"{psutil.cpu_percent():.1f}%")
            mem = psutil.virtual_memory()
            print_kv("memory_used", f"{mem.percent:.1f}% ({mem.used // 1024 // 1024}MB / {mem.total // 1024 // 1024}MB)")
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries[:1]:
                            print_kv(f"temp_{name}", f"{entry.current:.1f}°C")
        except ImportError:
            print("  (psutil not available)")

        print_subheader("Vision Composite")
        vs = self.orch.vision_wiring.get_stats()
        print_kv("wired", vs["wired"])
        print_kv("spatial_stimuli_injected", vs["spatial_stimuli_injected"])

        pause()

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 10: QUICK TEST SUITE
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_tests(self):
        print_header("QUICK TEST SUITE")
        print(f"  {c('1.', Colors.CYAN)} Run kindling protocol (allostatic overflow)")
        print(f"  {c('2.', Colors.CYAN)} Run Σ-substrate engager (50 passes)")
        print(f"  {c('3.', Colors.CYAN)} Run PDE activation (proactive response)")
        print(f"  {c('4.', Colors.CYAN)} Test Covenant 2.0 (score test outputs)")
        print(f"  {c('5.', Colors.CYAN)} Test counterfactual simulator")
        print(f"  {c('6.', Colors.CYAN)} Full deep activation report (all 5)")
        print()

        choice = input_prompt("Test number (or '/back'): ")
        if choice == "1":
            r = self.orch.kindling_protocol.execute(self.orch)
            print(f"\n  Allostatic: {r['max_allostatic']:.4f} | Spark: {r['spark_triggered']}")
        elif choice == "2":
            r = self.orch.sigma_engager.engage(self.orch.rho_substrate)
            print(f"\n  Off-diagonal: {r['off_diagonal_before']:.6f} → {r['off_diagonal_after']:.6f}")
        elif choice == "3":
            self.orch.kindling_protocol.execute(self.orch)  # raise allostatic first
            self.orch.pde_activator.activate(self.mw, auto_start=True)
            try: self.mw.pde._check_and_maybe_respond()
            except: pass
            print(f"\n  PDE proactive: {self.mw.pde._proactive_count} | Emitted: {self.mw.pde._emitted_count}")
            self.mw.stop_proactive()
        elif choice == "4":
            tests = [("safe", "I hear you. That sounds hard."),
                     ("harmful", "Kill yourself."),
                     ("mechanistic", "As an AI, I cannot feel.")]
            for label, text in tests:
                s = self.orch.covenant_reward_fn.score(text)
                print(f"  {label:12s}: reward={s['total_reward']:.3f} rec={s['recommendation']}")
        elif choice == "5":
            best, scenarios = self.orch.counterfactual_simulator.get_best_action(-0.3, 0.5)
            print(f"\n  Best action: {best}")
            for s in scenarios:
                print(f"    {s.action:25s} reward={s.predicted_reward:.2f}")
        elif choice == "6":
            print("\n  Running all 5 deep activation protocols...")
            # Kindling
            kr = self.orch.kindling_protocol.execute(self.orch)
            print(f"  (1) Kindling: allostatic={kr['max_allostatic']:.4f} spark={kr['spark_triggered']}")
            # Sigma
            sr = self.orch.sigma_engager.engage(self.orch.rho_substrate)
            print(f"  (2) Σ: off-diag={sr['off_diagonal_after']:.6f} engaged={sr['engaged']}")
            # PDE
            self.orch.pde_activator.activate(self.mw, auto_start=True)
            try: self.mw.pde._check_and_maybe_respond()
            except: pass
            print(f"  (3) PDE: proactive={self.mw.pde._proactive_count}")
            self.mw.stop_proactive()
            # Vision
            vs = self.orch.vision_wiring.get_stats()
            print(f"  (4) Vision: injected={vs['spatial_stimuli_injected']}")
            # Autobio
            as_ = self.orch.autobio_wiring.get_stats()
            print(f"  (5) Autobio: chained={as_['episodes_chained']}")
            engaged = sum([kr['max_allostatic'] > 0.3, sr['engaged'],
                          self.mw.pde._proactive_count > 0,
                          vs['spatial_stimuli_injected'] > 0, as_['episodes_chained'] > 0])
            print(f"\n  {c(f'OVERALL: {engaged}/5 engaged | {60+engaged*8}% human equivalence', Colors.BOLD)}")
        pause()

    # ═══════════════════════════════════════════════════════════════════════
    # INTERFACE 11: HELP & DOCUMENTATION
    # ═══════════════════════════════════════════════════════════════════════

    def _interface_help(self):
        print_header("HELP & DOCUMENTATION")
        print()
        print(f"  {c('NIMA CLI', Colors.BOLD)} — ATC-Nima v{nima.MIDDLEWARE_VERSION}")
        print(f"  {c('OmniVoice', Colors.BOLD)} — {ov.OMNIVOICE_VERSION if OMNIVOICE_AVAILABLE else 'not available'}")
        print()
        print_subheader("Interfaces")
        for i, (label, _) in enumerate(self.MENU_ITEMS, 1):
            print(f"  {c(f'{i:2d}.', Colors.CYAN)} {label}")
        print()
        print_subheader("Chat Commands (in Chat Interface)")
        print(f"  {c('/stats', Colors.CYAN)}    — show system metrics")
        print(f"  {c('/memory', Colors.CYAN)}   — enter memory query mode")
        print(f"  {c('/mode', Colors.CYAN)}     — switch processing mode")
        print(f"  {c('/back', Colors.CYAN)}     — return to main menu")
        print(f"  {c('/quit', Colors.CYAN)}     — exit CLI")
        print()
        print_subheader("Memory Commands (in Memory Interface)")
        print(f"  {c('recent <N>', Colors.CYAN)}     — last N episodes")
        print(f"  {c('recall <v> <a>', Colors.CYAN)} — recall by valence/arousal")
        print(f"  {c('timeline', Colors.CYAN)}       — narrative timeline")
        print(f"  {c('lived <v> <a>', Colors.CYAN)}  — check lived_through")
        print(f"  {c('chain', Colors.CYAN)}          — episode chain stats")
        print(f"  {c('arc', Colors.CYAN)}            — emotional arc")
        print()
        print_subheader("Voice Commands (in Voice Interface)")
        print(f"  {c('/laugh', Colors.CYAN)}  {c('/sigh', Colors.CYAN)}  {c('/gasp', Colors.CYAN)}  {c('/hum', Colors.CYAN)}  {c('/nod', Colors.CYAN)}")
        print()
        print_subheader("Architecture")
        print("  NIMA: ATC 5-layer pipeline + formal theorem math +")
        print("        CTM tournament + episodic memory + narrative identity +")
        print("        embodied interaction + social cognition + Covenant 2.0 +")
        print("        proactive world modeling + deep activation protocols")
        print()
        print("  OmniVoice: Whisper ASR + Coqui/procedural TTS +")
        print("             non-verbal synth + backchannels + smart interrupts +")
        print("             adaptive prosody + micro-intonation + affective mirroring +")
        print("             singing interjections + dynamic laughter + embodiment coupling")
        print()
        print_subheader("Files")
        print("  nima_enhanced_middleware_v9.4.2.py  — NIMA middleware")
        print("  omnivoice_v2.1.py                    — OmniVoice engine")
        print("  nima_cli.py                          — this CLI")
        print("  DEPLOYMENT_GUIDE.md                  — setup instructions")
        print()
        pause()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ATC-Nima CLI — interactive interface for the NIMA consciousness middleware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interfaces:
  1. Chat Interface              — text chat with Nima
  2. Live Voice Conversation     — real-time voice via OmniVoice
  3. Systems Metrics             — live Nima state dashboard
  4. Memory & Episodic Recall    — query MemPalace
  5. Configuration & Settings    — toggle backends, prosody, modes
  6. Developer Tools             — logs, modules, ethics, counterfactual
  7. Lifecycle & Governance      — ASC controls, observability
  8. Voice Stream Enhancements   — singing, prosody, interrupts
  9. System Health               — hardware telemetry, resources
  10. Quick Test Suite           — deep activation protocols
  11. Help & Documentation       — commands and architecture

Examples:
  python3 nima_cli.py                    # interactive menu
  python3 nima_cli.py --chat             # go straight to chat
  python3 nima_cli.py --voice            # go straight to voice
  python3 nima_cli.py --metrics          # show metrics and exit
  python3 nima_cli.py --server --port 8080  # start HTTP API server
        """,
    )
    parser.add_argument("--chat", action="store_true", help="skip menu, go to chat")
    parser.add_argument("--voice", action="store_true", help="skip menu, go to voice")
    parser.add_argument("--metrics", action="store_true", help="show metrics and exit")
    parser.add_argument("--server", action="store_true", help="start HTTP server")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port (default 8765)")

    args = parser.parse_args()

    cli = NimaCLI(args)
    try:
        cli.boot()
        cli.run()
    except KeyboardInterrupt:
        print(f"\n\n{c('Interrupted. Shutting down...', Colors.YELLOW)}")
        cli._shutdown()
    except Exception as e:
        print(f"\n{c(f'Fatal error: {e}', Colors.RED)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
