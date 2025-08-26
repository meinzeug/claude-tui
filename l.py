#!/usr/bin/env python3
"""
l.py - Live Datei-Monitor mit schöner TUI
Überwacht Dateierstellung und -änderungen im aktuellen Verzeichnis
"""

# ========== KONFIGURATION ==========
UPDATE_INTERVAL = 2  # Aktualisierungsintervall in Sekunden
# ====================================

import os
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import threading
from collections import deque

try:
    from rich.console import Console
    from rich.table import Table
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.align import Align
    from rich.text import Text
    from rich import box
except ImportError:
    print("Bitte installiere die 'rich' Bibliothek:")
    print("pip install rich")
    sys.exit(1)

class FileMonitor:
    def __init__(self, path="."):
        self.path = Path(path).resolve()
        self.console = Console()
        self.file_cache = {}
        self.recent_created = deque(maxlen=5)
        self.recent_modified = deque(maxlen=10)
        self.dir_sizes = {}
        self.last_scan = time.time()
        
    def format_size(self, size_bytes):
        """Formatiert Bytes in lesbare Größen"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def format_time(self, timestamp):
        """Formatiert Unix-Timestamp in lesbares Datum/Zeit"""
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%d.%m.%Y %H:%M:%S")
    
    def get_dir_size(self, path):
        """Berechnet die Größe eines Verzeichnisses in KB"""
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += self.get_dir_size(entry.path)
        except (PermissionError, OSError):
            pass
        return total / 1024  # Konvertiere zu KB
    
    def scan_directory(self):
        """Scannt das Verzeichnis nach Änderungen"""
        current_files = {}
        all_files = []
        
        # Sammle alle Dateien im aktuellen Verzeichnis
        for root, dirs, files in os.walk(self.path):
            # Alle Verzeichnisse einbeziehen (auch versteckte)
            for file in files:
                # Alle Dateien einbeziehen (auch versteckte)
                filepath = Path(root) / file
                try:
                    stat = filepath.stat()
                    file_info = {
                        'path': filepath,
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'ctime': stat.st_ctime,
                        'relative_path': filepath.relative_to(self.path)
                    }
                    current_files[str(filepath)] = file_info
                    all_files.append(file_info)
                except (PermissionError, OSError):
                    continue
        
        # Erkenne neue Dateien
        for filepath, info in current_files.items():
            if filepath not in self.file_cache:
                self.recent_created.append(info)
            elif self.file_cache[filepath]['mtime'] < info['mtime']:
                self.recent_modified.append(info)
        
        self.file_cache = current_files
        
        # Sortiere Dateien nach Erstellungszeit/Änderungszeit
        all_files.sort(key=lambda x: x['ctime'], reverse=True)
        newest_created = all_files[:5]
        
        all_files.sort(key=lambda x: x['mtime'], reverse=True)
        newest_modified = all_files[:10]
        
        # Berechne Verzeichnisgrößen
        self.dir_sizes = {}
        for item in os.listdir(self.path):
            item_path = self.path / item
            if item_path.is_dir():  # Alle Verzeichnisse einbeziehen
                self.dir_sizes[item] = self.get_dir_size(item_path)
        
        return newest_created, newest_modified
    
    def create_layout(self):
        """Erstellt das Layout für die TUI"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="recent_files", ratio=1),
            Layout(name="dir_sizes", ratio=1)
        )
        
        return layout
    
    def create_recent_files_table(self, files, title):
        """Erstellt eine Tabelle für kürzlich erstellte/geänderte Dateien"""
        table = Table(title=title, box=box.ROUNDED, show_header=True, 
                     header_style="bold cyan", title_style="bold white")
        table.add_column("Dateiname", style="green", width=30)
        table.add_column("Größe", justify="right", style="yellow")
        table.add_column("Datum/Zeit", style="blue")
        
        for file_info in files:
            name = str(file_info['relative_path'])
            if len(name) > 28:
                name = "..." + name[-25:]
            table.add_row(
                name,
                self.format_size(file_info['size']),
                self.format_time(file_info['mtime'])
            )
        
        return table
    
    def create_dir_sizes_table(self):
        """Erstellt eine Tabelle für Verzeichnisgrößen"""
        table = Table(title="📁 Verzeichnisgrößen", box=box.ROUNDED, 
                     show_header=True, header_style="bold cyan", title_style="bold white")
        table.add_column("Verzeichnis", style="magenta", width=25)
        table.add_column("Größe (KB)", justify="right", style="yellow")
        table.add_column("Größe", justify="right", style="green")
        
        sorted_dirs = sorted(self.dir_sizes.items(), key=lambda x: x[1], reverse=True)
        
        for dir_name, size_kb in sorted_dirs[:10]:  # Top 10 Verzeichnisse
            if len(dir_name) > 23:
                dir_name = dir_name[:20] + "..."
            table.add_row(
                dir_name,
                f"{size_kb:.1f}",
                self.format_size(size_kb * 1024)
            )
        
        return table
    
    def create_modified_files_table(self, files):
        """Erstellt eine Tabelle für zuletzt geänderte Dateien"""
        table = Table(title="📝 Zuletzt geänderte Dateien", box=box.ROUNDED,
                     show_header=True, header_style="bold cyan", title_style="bold white")
        table.add_column("Dateiname", style="cyan", width=35)
        table.add_column("Größe", justify="right", style="yellow")
        table.add_column("Geändert", style="blue")
        
        for file_info in files:
            name = str(file_info['relative_path'])
            if len(name) > 33:
                name = "..." + name[-30:]
            
            # Zeitunterschied berechnen
            time_diff = time.time() - file_info['mtime']
            if time_diff < 60:
                time_str = f"{int(time_diff)}s her"
            elif time_diff < 3600:
                time_str = f"{int(time_diff/60)}m her"
            elif time_diff < 86400:
                time_str = f"{int(time_diff/3600)}h her"
            else:
                time_str = self.format_time(file_info['mtime'])
            
            table.add_row(
                name,
                self.format_size(file_info['size']),
                time_str
            )
        
        return table
    
    def generate_display(self):
        """Generiert die komplette Anzeige"""
        newest_created, newest_modified = self.scan_directory()
        layout = self.create_layout()
        
        # Header
        header_text = Text()
        header_text.append("🔍 Live Datei-Monitor", style="bold white")
        header_text.append(f" - {self.path}", style="dim white")
        header_text.append(f"\n📂 {len(self.file_cache)} Dateien überwacht", style="green")
        header_text.append(f" | ⏱️  Aktualisierung alle {UPDATE_INTERVAL} Sekunden", style="yellow")
        layout["header"].update(Panel(Align.center(header_text), box=box.ROUNDED))
        
        # Zuletzt erstellte Dateien
        layout["recent_files"].update(
            Panel(self.create_recent_files_table(newest_created, "🆕 Neueste Dateien"), 
                  box=box.ROUNDED)
        )
        
        # Verzeichnisgrößen
        layout["dir_sizes"].update(
            Panel(self.create_dir_sizes_table(), box=box.ROUNDED)
        )
        
        # Zuletzt geänderte Dateien
        layout["right"].update(
            Panel(self.create_modified_files_table(newest_modified), box=box.ROUNDED)
        )
        
        # Footer
        footer_text = Text()
        footer_text.append("ESC/Q: Beenden", style="bold red")
        footer_text.append(" | ", style="dim white")
        footer_text.append(f"Letzte Aktualisierung: {datetime.now().strftime('%H:%M:%S')}", 
                         style="blue")
        layout["footer"].update(Panel(Align.center(footer_text), box=box.ROUNDED))
        
        return layout
    
    def run(self):
        """Hauptschleife des Monitors"""
        try:
            # Initialer Scan
            self.scan_directory()
            
            with Live(self.generate_display(), refresh_per_second=1, 
                     screen=True) as live:
                while True:
                    try:
                        # Aktualisiere nach dem festgelegten Intervall
                        time.sleep(UPDATE_INTERVAL)
                        live.update(self.generate_display())
                        
                    except KeyboardInterrupt:
                        break
                        
        except KeyboardInterrupt:
            pass
        finally:
            self.console.print("\n[bold red]Monitor beendet.[/bold red]")

def main():
    """Hauptfunktion"""
    console = Console()
    
    # Banner
    console.print(Panel.fit(
        "[bold cyan]📊 Live Datei-Monitor[/bold cyan]\n"
        "[dim]Überwacht Dateien im aktuellen Verzeichnis[/dim]",
        box=box.DOUBLE
    ))
    
    # Starte den Monitor
    monitor = FileMonitor()
    console.print(f"[green]Überwache:[/green] {monitor.path}\n")
    console.print("[yellow]Drücke Strg+C zum Beenden...[/yellow]\n")
    time.sleep(2)
    
    monitor.run()

if __name__ == "__main__":
    main()
