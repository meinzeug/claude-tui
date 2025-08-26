"""
Marketplace UI Components - Comprehensive marketplace widgets for Claude-TUI.

Features:
- Real-time marketplace browsing
- Plugin installation interface
- Template gallery with preview
- Rating and review display
- Search and filtering interface
- Purchase flow for premium items
- Installation progress tracking
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.console import RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button, Collapsible, DataTable, Footer, Header, Input, Label, 
    LoadingIndicator, Markdown, OptionList, Placeholder, SelectionList,
    Static, TabPane, TabbedContent, Tree
)

from ...api.client import APIClient
from ...core.config import Config
from ...integrations.websocket_client import WebSocketClient


class MarketplaceSearchBar(Container):
    """Advanced search bar with filters and autocomplete for marketplace browsing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.search_query = ""
        self.active_filters = {}
        self.autocomplete_suggestions = []
        self.api_client = APIClient()

    def compose(self) -> ComposeResult:
        """Create the search bar interface."""
        with Horizontal(classes="search-container"):
            yield Input(
                placeholder="Search templates, plugins, and extensions...",
                id="search-input",
                classes="search-input"
            )
            yield Button("🔍", id="search-button", classes="search-button")
            yield Button("🎚️", id="filters-button", classes="filters-button")
            yield Button("📈", id="trending-button", classes="trending-button")

        # Autocomplete suggestions dropdown
        yield Container(id="autocomplete-suggestions", classes="autocomplete-container")

        with Collapsible(title="Advanced Filters", id="filters-panel", collapsed=True):
            with Horizontal(classes="filter-row"):
                yield SelectionList(
                    "Templates", "Plugins", "Extensions", "Themes",
                    id="type-filter",
                    classes="filter-selection"
                )
                yield SelectionList(
                    "Python", "JavaScript", "Rust", "Go", "TypeScript",
                    id="language-filter", 
                    classes="filter-selection"
                )

            with Horizontal(classes="filter-row"):
                yield SelectionList(
                    "Free", "Premium", "Open Source", "Commercial",
                    id="pricing-filter",
                    classes="filter-selection"
                )
                yield SelectionList(
                    "Beginner", "Intermediate", "Advanced", "Expert",
                    id="complexity-filter",
                    classes="filter-selection"
                )

            with Horizontal(classes="filter-row"):
                yield SelectionList(
                    "⭐ 5 stars", "⭐ 4+ stars", "⭐ 3+ stars", "⭐ 2+ stars",
                    id="rating-filter",
                    classes="filter-selection"
                )
                yield SelectionList(
                    "Last 24h", "Last week", "Last month", "Last year",
                    id="updated-filter",
                    classes="filter-selection"
                )

    @on(Input.Changed, "#search-input")
    async def handle_search_changed(self, event: Input.Changed) -> None:
        """Handle search input changes for autocomplete."""
        if len(event.value) >= 2:  # Start autocomplete after 2 characters
            await self.get_autocomplete_suggestions(event.value)
        else:
            await self.hide_autocomplete_suggestions()

    @on(Input.Submitted, "#search-input")
    async def handle_search_submit(self, event: Input.Submitted) -> None:
        """Handle search submission."""
        self.search_query = event.value
        await self.hide_autocomplete_suggestions()
        await self.parent.perform_search(self.search_query, self.active_filters)

    @on(Button.Pressed, "#search-button")
    async def handle_search_button(self, event: Button.Pressed) -> None:
        """Handle search button click."""
        search_input = self.query_one("#search-input", Input)
        self.search_query = search_input.value
        await self.hide_autocomplete_suggestions()
        await self.parent.perform_search(self.search_query, self.active_filters)

    @on(Button.Pressed, "#trending-button")
    async def handle_trending_button(self, event: Button.Pressed) -> None:
        """Handle trending searches button click."""
        await self.show_trending_searches()

    @on(Button.Pressed, "#filters-button")
    async def toggle_filters(self, event: Button.Pressed) -> None:
        """Toggle filters panel visibility."""
        filters_panel = self.query_one("#filters-panel", Collapsible)
        filters_panel.collapsed = not filters_panel.collapsed

    async def get_autocomplete_suggestions(self, query: str) -> None:
        """Get autocomplete suggestions from API."""
        try:
            response = await self.api_client.get(f"/marketplace/search/autocomplete?q={query}")
            suggestions = response.get("suggestions", [])
            await self.show_autocomplete_suggestions(suggestions)
        except Exception as e:
            # Silently handle autocomplete errors to avoid interrupting user experience
            await self.hide_autocomplete_suggestions()

    async def show_autocomplete_suggestions(self, suggestions: List[Dict[str, Any]]) -> None:
        """Show autocomplete suggestions dropdown."""
        suggestions_container = self.query_one("#autocomplete-suggestions", Container)
        suggestions_container.remove_children()
        
        if suggestions:
            for suggestion in suggestions[:8]:  # Limit to 8 suggestions
                suggestion_text = suggestion.get("text", "")
                suggestion_type = suggestion.get("type", "")
                suggestion_count = suggestion.get("count", 0)
                
                suggestion_widget = Button(
                    f"{suggestion_text} ({suggestion_count})",
                    classes=f"autocomplete-item autocomplete-{suggestion_type}",
                    id=f"suggestion-{suggestion_text}"
                )
                suggestions_container.mount(suggestion_widget)
            
            suggestions_container.styles.display = "block"
        else:
            await self.hide_autocomplete_suggestions()

    async def hide_autocomplete_suggestions(self) -> None:
        """Hide autocomplete suggestions dropdown."""
        suggestions_container = self.query_one("#autocomplete-suggestions", Container)
        suggestions_container.styles.display = "none"
        suggestions_container.remove_children()

    async def show_trending_searches(self) -> None:
        """Show trending searches popup."""
        try:
            response = await self.api_client.get("/marketplace/search/trending")
            trending = response.get("trending_searches", [])
            
            # Create trending searches widget
            trending_widget = Container(classes="trending-popup")
            trending_widget.mount(Label("🔥 Trending Searches", classes="trending-title"))
            
            for trend in trending[:10]:
                term = trend.get("term", "")
                count = trend.get("count", 0)
                trend_direction = trend.get("trend", "stable")
                
                trend_icon = "📈" if trend_direction == "up" else "📉" if trend_direction == "down" else "➖"
                trend_button = Button(
                    f"{trend_icon} {term} ({count})",
                    classes="trending-item",
                    id=f"trending-{term}"
                )
                trending_widget.mount(trend_button)
            
            # Show popup (in a real implementation, you'd use a modal or overlay)
            suggestions_container = self.query_one("#autocomplete-suggestions", Container)
            suggestions_container.remove_children()
            suggestions_container.mount(trending_widget)
            suggestions_container.styles.display = "block"
            
        except Exception as e:
            # Handle trending search errors gracefully
            pass

    @on(Button.Pressed, "[id^='suggestion-']")
    async def handle_suggestion_click(self, event: Button.Pressed) -> None:
        """Handle autocomplete suggestion click."""
        suggestion_text = event.button.id.replace("suggestion-", "")
        search_input = self.query_one("#search-input", Input)
        search_input.value = suggestion_text
        self.search_query = suggestion_text
        await self.hide_autocomplete_suggestions()
        await self.parent.perform_search(self.search_query, self.active_filters)

    @on(Button.Pressed, "[id^='trending-']")
    async def handle_trending_click(self, event: Button.Pressed) -> None:
        """Handle trending search click."""
        trending_term = event.button.id.replace("trending-", "")
        search_input = self.query_one("#search-input", Input)
        search_input.value = trending_term
        self.search_query = trending_term
        await self.hide_autocomplete_suggestions()
        await self.parent.perform_search(self.search_query, self.active_filters)

    @on(SelectionList.SelectedChanged)
    async def handle_filter_change(self, event: SelectionList.SelectedChanged) -> None:
        """Handle filter selection changes."""
        filter_name = event.control.id.replace("-filter", "")
        self.active_filters[filter_name] = [
            item for item in event.control.selected
        ]
        await self.parent.perform_search(self.search_query, self.active_filters)


class MarketplaceItemCard(Container):
    """Individual marketplace item display card."""

    def __init__(self, item_data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.item_data = item_data
        self.item_id = item_data.get("id")
        self.item_type = item_data.get("type", "template")

    def compose(self) -> ComposeResult:
        """Create item card interface."""
        with Container(classes="item-card"):
            # Header with icon and title
            with Horizontal(classes="item-header"):
                yield Static(
                    self._get_type_icon(),
                    classes="item-icon"
                )
                with Vertical(classes="item-title-section"):
                    yield Label(
                        self.item_data.get("name", "Unknown"),
                        classes="item-title"
                    )
                    yield Label(
                        f"by {self.item_data.get('author', 'Unknown')}",
                        classes="item-author"
                    )

            # Description
            yield Static(
                self.item_data.get("short_description", "No description available"),
                classes="item-description"
            )

            # Tags and categories
            if self.item_data.get("tags"):
                tag_text = " ".join([f"#{tag}" for tag in self.item_data["tags"][:5]])
                yield Static(tag_text, classes="item-tags")

            # Stats and actions
            with Horizontal(classes="item-footer"):
                with Vertical(classes="item-stats"):
                    stats_text = self._format_stats()
                    yield Static(stats_text, classes="stats-text")

                with Vertical(classes="item-actions"):
                    if self.item_data.get("is_premium"):
                        price = self.item_data.get("price", 0)
                        yield Button(f"Buy ${price:.2f}", id="buy-button", variant="primary")
                    else:
                        if self.item_type == "plugin":
                            yield Button("Install", id="install-button", variant="success")
                        else:
                            yield Button("Download", id="download-button", variant="success")
                    
                    yield Button("Preview", id="preview-button", variant="default")

    def _get_type_icon(self) -> str:
        """Get appropriate icon for item type."""
        icons = {
            "template": "📄",
            "plugin": "🧩",
            "extension": "⚡",
            "theme": "🎨"
        }
        return icons.get(self.item_type, "📦")

    def _format_stats(self) -> str:
        """Format item statistics."""
        stats = []
        
        if downloads := self.item_data.get("download_count"):
            stats.append(f"📥 {downloads:,}")
        
        if rating := self.item_data.get("average_rating"):
            stars = "⭐" * int(rating)
            stats.append(f"{stars} {rating:.1f}")
        
        if updated := self.item_data.get("updated_at"):
            # Parse and format date
            stats.append(f"Updated: {updated[:10]}")

        return " | ".join(stats)

    @on(Button.Pressed, "#install-button")
    async def handle_install(self, event: Button.Pressed) -> None:
        """Handle plugin installation."""
        await self.parent.install_plugin(self.item_id)

    @on(Button.Pressed, "#download-button")
    async def handle_download(self, event: Button.Pressed) -> None:
        """Handle template download."""
        await self.parent.download_template(self.item_id)

    @on(Button.Pressed, "#preview-button")
    async def handle_preview(self, event: Button.Pressed) -> None:
        """Handle item preview."""
        await self.parent.show_item_preview(self.item_id, self.item_type)

    @on(Button.Pressed, "#buy-button")
    async def handle_purchase(self, event: Button.Pressed) -> None:
        """Handle premium item purchase."""
        await self.parent.purchase_item(self.item_id)


class MarketplaceGrid(ScrollableContainer):
    """Grid layout for marketplace items with infinite scrolling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.items = []
        self.current_page = 1
        self.loading = False
        self.has_more = True

    def compose(self) -> ComposeResult:
        """Create grid interface."""
        yield Container(id="items-container", classes="marketplace-grid")
        yield LoadingIndicator(id="loading-indicator")

    async def load_items(self, search_query: str = "", filters: Dict[str, Any] = None):
        """Load marketplace items."""
        if self.loading:
            return

        self.loading = True
        loading_indicator = self.query_one("#loading-indicator", LoadingIndicator)
        loading_indicator.display = True

        try:
            api_client = APIClient()
            response = await api_client.get("/marketplace/search", params={
                "query": search_query,
                "page": self.current_page,
                "page_size": 20,
                **(filters or {})
            })

            if response.get("items"):
                new_items = response["items"]
                self.items.extend(new_items)
                
                # Add new item cards to grid
                items_container = self.query_one("#items-container")
                for item in new_items:
                    card = MarketplaceItemCard(item)
                    await items_container.mount(card)
                
                self.current_page += 1
                self.has_more = len(new_items) == 20
            else:
                self.has_more = False

        except Exception as e:
            self.notify(f"Failed to load marketplace items: {str(e)}", severity="error")
        finally:
            self.loading = False
            loading_indicator.display = False

    async def clear_items(self):
        """Clear all items from grid."""
        items_container = self.query_one("#items-container")
        await items_container.remove_children()
        self.items.clear()
        self.current_page = 1
        self.has_more = True

    async def perform_search(self, query: str, filters: Dict[str, Any]):
        """Perform new search with query and filters."""
        await self.clear_items()
        await self.load_items(query, filters)

    async def install_plugin(self, plugin_id: str):
        """Install a plugin."""
        try:
            api_client = APIClient()
            response = await api_client.post("/marketplace/plugins/install", json={
                "plugin_id": plugin_id,
                "installation_method": "marketplace"
            })
            
            installation_id = response.get("installation_id")
            if installation_id:
                await self.show_installation_progress(installation_id)
            
            self.notify(f"Plugin installation started", severity="success")
            
        except Exception as e:
            self.notify(f"Installation failed: {str(e)}", severity="error")

    async def download_template(self, template_id: str):
        """Download a template."""
        try:
            api_client = APIClient()
            await api_client.post(f"/marketplace/templates/{template_id}/download")
            self.notify("Template downloaded successfully", severity="success")
            
        except Exception as e:
            self.notify(f"Download failed: {str(e)}", severity="error")

    async def show_item_preview(self, item_id: str, item_type: str):
        """Show item preview dialog."""
        await self.parent.show_preview_dialog(item_id, item_type)

    async def purchase_item(self, item_id: str):
        """Purchase premium item."""
        await self.parent.show_purchase_dialog(item_id)

    @work(exclusive=True)
    async def show_installation_progress(self, installation_id: str):
        """Show installation progress in real-time."""
        progress_dialog = InstallationProgressDialog(installation_id)
        await self.parent.push_screen(progress_dialog)


class InstallationProgressDialog(Container):
    """Dialog showing real-time plugin installation progress."""

    def __init__(self, installation_id: str, **kwargs):
        super().__init__(**kwargs)
        self.installation_id = installation_id
        self.ws_client = None

    def compose(self) -> ComposeResult:
        """Create progress dialog interface."""
        with Container(classes="progress-dialog"):
            yield Label("Installing Plugin...", classes="dialog-title")
            
            with Container(classes="progress-container"):
                yield Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=40),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    id="installation-progress"
                )
            
            yield Static("Initializing installation...", id="status-message")
            
            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button("Background", id="background-button", variant="default")

    async def on_mount(self) -> None:
        """Start monitoring installation progress."""
        await self.start_progress_monitoring()

    @work(exclusive=True)
    async def start_progress_monitoring(self):
        """Monitor installation progress via WebSocket."""
        try:
            self.ws_client = WebSocketClient()
            await self.ws_client.connect("/marketplace/ws/updates")
            
            # Subscribe to installation updates
            await self.ws_client.send({
                "type": "subscribe",
                "channel": f"installation:{self.installation_id}"
            })
            
            progress = self.query_one("#installation-progress", Progress)
            status_message = self.query_one("#status-message", Static)
            
            task_id = progress.add_task("Installing...", total=100)
            
            async for message in self.ws_client.listen():
                data = json.loads(message)
                
                if data.get("installation_id") == self.installation_id:
                    progress_value = data.get("progress", 0)
                    message_text = data.get("message", "")
                    status = data.get("status", "")
                    
                    progress.update(task_id, completed=progress_value)
                    status_message.update(message_text)
                    
                    if status == "completed":
                        status_message.update("✅ Installation completed successfully!")
                        await asyncio.sleep(2)
                        await self.dismiss()
                        break
                    elif status == "failed":
                        error_msg = data.get("error_message", "Unknown error")
                        status_message.update(f"❌ Installation failed: {error_msg}")
                        break
                        
        except Exception as e:
            status_message = self.query_one("#status-message", Static)
            status_message.update(f"❌ Error monitoring installation: {str(e)}")

    @on(Button.Pressed, "#cancel-button")
    async def cancel_installation(self, event: Button.Pressed) -> None:
        """Cancel the installation."""
        try:
            api_client = APIClient()
            await api_client.delete(f"/marketplace/installations/{self.installation_id}")
            await self.dismiss()
        except Exception as e:
            self.notify(f"Failed to cancel installation: {str(e)}", severity="error")

    @on(Button.Pressed, "#background-button")
    async def run_in_background(self, event: Button.Pressed) -> None:
        """Continue installation in background."""
        await self.dismiss()

    async def dismiss(self) -> None:
        """Close the dialog."""
        if self.ws_client:
            await self.ws_client.disconnect()
        await self.parent.pop_screen()


class MarketplacePreviewDialog(Container):
    """Dialog for previewing marketplace items."""

    def __init__(self, item_id: str, item_type: str, **kwargs):
        super().__init__(**kwargs)
        self.item_id = item_id
        self.item_type = item_type
        self.item_data = None

    def compose(self) -> ComposeResult:
        """Create preview dialog interface."""
        with Container(classes="preview-dialog"):
            yield Header(id="preview-header")
            
            with TabbedContent(id="preview-tabs"):
                with TabPane("Overview", id="overview-tab"):
                    yield ScrollableContainer(
                        Markdown("Loading..."),
                        id="overview-content"
                    )
                
                with TabPane("Files", id="files-tab"):
                    yield Tree("Files", id="files-tree")
                
                with TabPane("Reviews", id="reviews-tab"):
                    yield ScrollableContainer(id="reviews-content")
                
                if self.item_type == "plugin":
                    with TabPane("Permissions", id="permissions-tab"):
                        yield ScrollableContainer(id="permissions-content")

            with Horizontal(classes="preview-actions"):
                if self.item_type == "plugin":
                    yield Button("Install Plugin", id="install-action", variant="success")
                else:
                    yield Button("Download Template", id="download-action", variant="success")
                
                yield Button("View on GitHub", id="github-action", variant="default")
                yield Button("Report Issue", id="report-action", variant="warning")
                yield Button("Close", id="close-action", variant="default")

    async def on_mount(self) -> None:
        """Load item details when dialog mounts."""
        await self.load_item_details()

    @work(exclusive=True)
    async def load_item_details(self):
        """Load detailed item information."""
        try:
            api_client = APIClient()
            
            # Load main item data
            if self.item_type == "plugin":
                response = await api_client.get(f"/marketplace/plugins/{self.item_id}")
            else:
                response = await api_client.get(f"/marketplace/templates/{self.item_id}")
            
            self.item_data = response
            
            # Update header
            header = self.query_one("#preview-header", Header)
            header.title = self.item_data.get("name", "Unknown Item")
            
            # Update overview tab
            await self.update_overview_tab()
            
            # Load files structure
            await self.load_files_structure()
            
            # Load reviews
            await self.load_reviews()
            
            # Load permissions (for plugins)
            if self.item_type == "plugin":
                await self.load_permissions()
                
        except Exception as e:
            self.notify(f"Failed to load item details: {str(e)}", severity="error")

    async def update_overview_tab(self):
        """Update the overview tab with item details."""
        if not self.item_data:
            return

        overview_content = self.query_one("#overview-content", ScrollableContainer)
        
        # Create detailed overview markdown
        overview_md = f"""
# {self.item_data.get('name', 'Unknown')}

**Author:** {self.item_data.get('author', 'Unknown')}  
**Version:** {self.item_data.get('version', 'N/A')}  
**Type:** {self.item_type.title()}  
**License:** {self.item_data.get('license', 'N/A')}

## Description

{self.item_data.get('description', 'No description available.')}

## Features

{chr(10).join([f'- {feature}' for feature in self.item_data.get('features', [])])}

## Categories

{', '.join(self.item_data.get('categories', []))}

## Languages/Frameworks

{', '.join(self.item_data.get('languages', []))}

## Statistics

- **Downloads:** {self.item_data.get('download_count', 0):,}
- **Stars:** {self.item_data.get('star_count', 0):,}
- **Rating:** {self.item_data.get('average_rating', 0):.1f}/5.0
- **Last Updated:** {self.item_data.get('updated_at', 'Unknown')[:10]}
        """
        
        await overview_content.remove_children()
        await overview_content.mount(Markdown(overview_md))

    async def load_files_structure(self):
        """Load and display files structure."""
        try:
            api_client = APIClient()
            files_response = await api_client.get(
                f"/marketplace/{self.item_type}s/{self.item_id}/files"
            )
            
            files_tree = self.query_one("#files-tree", Tree)
            
            # Build tree structure
            if files_response.get("files"):
                for file_path in files_response["files"]:
                    # Add file to tree (simplified)
                    files_tree.root.add_leaf(file_path)
                    
        except Exception as e:
            self.notify(f"Failed to load files: {str(e)}", severity="warning")

    async def load_reviews(self):
        """Load and display reviews."""
        try:
            api_client = APIClient()
            reviews_response = await api_client.get(
                f"/marketplace/ratings/{self.item_id}?item_type={self.item_type}"
            )
            
            reviews_content = self.query_one("#reviews-content", ScrollableContainer)
            
            if reviews := reviews_response.get("ratings"):
                for review in reviews[:10]:  # Show top 10 reviews
                    review_widget = self.create_review_widget(review)
                    await reviews_content.mount(review_widget)
            else:
                await reviews_content.mount(
                    Static("No reviews available yet.", classes="no-reviews")
                )
                
        except Exception as e:
            self.notify(f"Failed to load reviews: {str(e)}", severity="warning")

    def create_review_widget(self, review_data: Dict[str, Any]) -> Container:
        """Create a review display widget."""
        with Container(classes="review-item") as review_container:
            # Review header
            with Horizontal(classes="review-header"):
                Static(
                    f"⭐ {review_data.get('rating', 0)}/5", 
                    classes="review-rating"
                )
                Static(
                    review_data.get('reviewer_name', 'Anonymous'),
                    classes="review-author"
                )
                Static(
                    review_data.get('created_at', '')[:10],
                    classes="review-date"
                )
            
            # Review content
            if title := review_data.get('review_title'):
                Static(title, classes="review-title")
            
            if text := review_data.get('review_text'):
                Static(text, classes="review-text")
        
        return review_container

    async def load_permissions(self):
        """Load and display plugin permissions."""
        if self.item_type != "plugin":
            return

        try:
            permissions_content = self.query_one("#permissions-content", ScrollableContainer)
            
            required_perms = self.item_data.get('required_permissions', [])
            optional_perms = self.item_data.get('optional_permissions', [])
            
            if required_perms or optional_perms:
                perms_table = Table(title="Plugin Permissions")
                perms_table.add_column("Permission")
                perms_table.add_column("Type")
                perms_table.add_column("Description")
                
                for perm in required_perms:
                    perms_table.add_row(
                        perm, 
                        "Required", 
                        self.get_permission_description(perm)
                    )
                
                for perm in optional_perms:
                    perms_table.add_row(
                        perm,
                        "Optional",
                        self.get_permission_description(perm)
                    )
                
                await permissions_content.mount(Static(perms_table))
            else:
                await permissions_content.mount(
                    Static("This plugin requires no special permissions.", 
                           classes="no-permissions")
                )
                
        except Exception as e:
            self.notify(f"Failed to load permissions: {str(e)}", severity="warning")

    def get_permission_description(self, permission: str) -> str:
        """Get human-readable permission description."""
        descriptions = {
            "file_system": "Read/write files on your system",
            "network": "Make network requests",
            "clipboard": "Access system clipboard",
            "notifications": "Show system notifications",
            "terminal": "Execute terminal commands"
        }
        return descriptions.get(permission, "Custom permission")

    @on(Button.Pressed, "#close-action")
    async def close_preview(self, event: Button.Pressed) -> None:
        """Close the preview dialog."""
        await self.dismiss()

    async def dismiss(self) -> None:
        """Close the preview dialog."""
        await self.parent.pop_screen()


class MarketplaceBrowser(Container):
    """Main marketplace browser interface."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_client = APIClient()

    def compose(self) -> ComposeResult:
        """Create marketplace browser interface."""
        yield Header(show_clock=True, name="Claude-TUI Marketplace")
        
        with Container(classes="marketplace-main"):
            yield MarketplaceSearchBar(id="search-bar")
            
            with Horizontal(classes="marketplace-content"):
                # Sidebar with categories and filters
                with Container(classes="marketplace-sidebar"):
                    yield Label("Categories", classes="sidebar-title")
                    yield SelectionList(
                        "All Items", "Templates", "Plugins", "Extensions", "Themes",
                        id="category-list"
                    )
                    
                    yield Label("Sort By", classes="sidebar-title")
                    yield OptionList(
                        "Most Popular", "Recently Updated", "Highest Rated", 
                        "Most Downloaded", "Newest",
                        id="sort-options"
                    )

                # Main content area
                yield MarketplaceGrid(id="marketplace-grid", classes="marketplace-main-content")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize marketplace browser."""
        # Load initial items
        marketplace_grid = self.query_one("#marketplace-grid", MarketplaceGrid)
        await marketplace_grid.load_items()

    async def show_preview_dialog(self, item_id: str, item_type: str):
        """Show item preview dialog."""
        preview_dialog = MarketplacePreviewDialog(item_id, item_type)
        await self.push_screen(preview_dialog)

    async def show_purchase_dialog(self, item_id: str):
        """Show purchase dialog for premium items."""
        # Implementation for purchase flow
        self.notify("Purchase flow not yet implemented", severity="info")