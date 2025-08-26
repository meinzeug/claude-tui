"""
Monetization Service - Premium features, subscriptions, and payment processing.

Features:
- Flexible pricing models (one-time, subscription, usage-based)
- Premium plugin/template distribution
- Tiered access control
- Payment processing integration
- Revenue sharing with creators
- Analytics and reporting
- Trial periods and promotions
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from sqlalchemy import desc, func, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)


class PricingEngine:
    """Dynamic pricing engine with multiple models."""
    
    def __init__(self):
        self.pricing_models = {
            "free": self._calculate_free_pricing,
            "one_time": self._calculate_one_time_pricing,
            "subscription": self._calculate_subscription_pricing,
            "usage_based": self._calculate_usage_pricing,
            "tiered": self._calculate_tiered_pricing,
            "freemium": self._calculate_freemium_pricing
        }
    
    async def calculate_price(
        self,
        item_data: Dict[str, Any],
        user_data: Optional[Dict[str, Any]] = None,
        usage_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Calculate price for an item based on various factors."""
        try:
            pricing_model = item_data.get("pricing_model", "free")
            
            if pricing_model not in self.pricing_models:
                raise ValueError(f"Unknown pricing model: {pricing_model}")
            
            calculator = self.pricing_models[pricing_model]
            price_info = await calculator(item_data, user_data, usage_data)
            
            # Apply discounts and promotions
            price_info = await self._apply_discounts(price_info, user_data)
            
            return price_info
            
        except Exception as e:
            logger.error(f"Error calculating price: {e}")
            return {"price": 0.0, "currency": "USD", "model": "free"}
    
    async def _calculate_free_pricing(
        self, item_data: Dict[str, Any], user_data: Optional[Dict[str, Any]], usage_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate free pricing."""
        return {
            "price": 0.0,
            "currency": "USD",
            "model": "free",
            "description": "Free to use"
        }
    
    async def _calculate_one_time_pricing(
        self, item_data: Dict[str, Any], user_data: Optional[Dict[str, Any]], usage_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate one-time purchase pricing."""
        base_price = Decimal(str(item_data.get("base_price", 0)))
        
        return {
            "price": float(base_price),
            "currency": item_data.get("currency", "USD"),
            "model": "one_time",
            "description": f"One-time purchase: ${base_price}"
        }
    
    async def _calculate_subscription_pricing(
        self, item_data: Dict[str, Any], user_data: Optional[Dict[str, Any]], usage_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate subscription pricing."""
        monthly_price = Decimal(str(item_data.get("monthly_price", 0)))
        annual_price = Decimal(str(item_data.get("annual_price", monthly_price * 10)))  # 2 months free
        
        return {
            "price": float(monthly_price),
            "annual_price": float(annual_price),
            "currency": item_data.get("currency", "USD"),
            "model": "subscription",
            "billing_cycle": "monthly",
            "description": f"${monthly_price}/month or ${annual_price}/year"
        }
    
    async def _calculate_usage_pricing(
        self, item_data: Dict[str, Any], user_data: Optional[Dict[str, Any]], usage_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate usage-based pricing."""
        usage_count = usage_data.get("usage_count", 0) if usage_data else 0
        price_per_use = Decimal(str(item_data.get("price_per_use", 0.1)))
        
        total_price = price_per_use * usage_count
        
        return {
            "price": float(total_price),
            "price_per_use": float(price_per_use),
            "usage_count": usage_count,
            "currency": item_data.get("currency", "USD"),
            "model": "usage_based",
            "description": f"${price_per_use} per use (${total_price} total)"
        }
    
    async def _calculate_tiered_pricing(
        self, item_data: Dict[str, Any], user_data: Optional[Dict[str, Any]], usage_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate tiered pricing based on user level or usage."""
        tiers = item_data.get("pricing_tiers", [])
        user_tier = user_data.get("subscription_tier", "basic") if user_data else "basic"
        
        # Find appropriate tier
        tier_price = 0.0
        for tier in tiers:
            if tier.get("tier_name") == user_tier:
                tier_price = tier.get("price", 0.0)
                break
        
        return {
            "price": tier_price,
            "tier": user_tier,
            "currency": item_data.get("currency", "USD"),
            "model": "tiered",
            "description": f"{user_tier.title()} tier: ${tier_price}"
        }
    
    async def _calculate_freemium_pricing(
        self, item_data: Dict[str, Any], user_data: Optional[Dict[str, Any]], usage_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate freemium pricing with usage limits."""
        free_limit = item_data.get("free_usage_limit", 10)
        premium_price = Decimal(str(item_data.get("premium_price", 9.99)))
        current_usage = usage_data.get("usage_count", 0) if usage_data else 0
        
        if current_usage < free_limit:
            return {
                "price": 0.0,
                "remaining_free": free_limit - current_usage,
                "upgrade_price": float(premium_price),
                "currency": item_data.get("currency", "USD"),
                "model": "freemium",
                "description": f"Free ({free_limit - current_usage} uses remaining), upgrade for ${premium_price}"
            }
        else:
            return {
                "price": float(premium_price),
                "currency": item_data.get("currency", "USD"),
                "model": "freemium",
                "description": f"Premium upgrade required: ${premium_price}"
            }
    
    async def _apply_discounts(
        self, price_info: Dict[str, Any], user_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply applicable discounts and promotions."""
        if not user_data:
            return price_info
        
        original_price = price_info.get("price", 0.0)
        discount_percent = 0
        discount_reason = None
        
        # Student discount
        if user_data.get("is_student"):
            discount_percent = max(discount_percent, 20)
            discount_reason = "Student discount"
        
        # First-time buyer discount
        if user_data.get("first_time_buyer"):
            discount_percent = max(discount_percent, 15)
            discount_reason = "First-time buyer discount"
        
        # Loyalty discount
        purchase_count = user_data.get("purchase_count", 0)
        if purchase_count >= 10:
            discount_percent = max(discount_percent, 25)
            discount_reason = "Loyalty discount"
        elif purchase_count >= 5:
            discount_percent = max(discount_percent, 10)
            discount_reason = "Returning customer discount"
        
        # Apply discount
        if discount_percent > 0:
            discount_amount = original_price * (discount_percent / 100)
            final_price = original_price - discount_amount
            
            price_info.update({
                "original_price": original_price,
                "price": final_price,
                "discount_percent": discount_percent,
                "discount_amount": discount_amount,
                "discount_reason": discount_reason
            })
        
        return price_info


class PaymentProcessor:
    """Payment processing integration."""
    
    def __init__(self):
        self.supported_methods = ["card", "paypal", "crypto", "bank_transfer"]
        self.currencies = ["USD", "EUR", "GBP", "JPY", "CAD"]
    
    async def create_payment_intent(
        self,
        amount: float,
        currency: str,
        customer_id: UUID,
        item_id: UUID,
        payment_method: str = "card"
    ) -> Dict[str, Any]:
        """Create a payment intent for processing."""
        try:
            if payment_method not in self.supported_methods:
                raise ValueError(f"Unsupported payment method: {payment_method}")
            
            if currency not in self.currencies:
                raise ValueError(f"Unsupported currency: {currency}")
            
            # Create payment intent (mock implementation)
            payment_intent = {
                "id": str(uuid4()),
                "amount": amount,
                "currency": currency,
                "customer_id": str(customer_id),
                "item_id": str(item_id),
                "payment_method": payment_method,
                "status": "requires_payment_method",
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat()
            }
            
            return payment_intent
            
        except Exception as e:
            logger.error(f"Error creating payment intent: {e}")
            raise
    
    async def confirm_payment(
        self,
        payment_intent_id: str,
        payment_method_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Confirm and process payment."""
        try:
            # Mock payment confirmation
            # In real implementation, this would integrate with Stripe, PayPal, etc.
            
            return {
                "payment_intent_id": payment_intent_id,
                "status": "succeeded",
                "amount_received": payment_method_data.get("amount", 0),
                "payment_method": payment_method_data.get("type", "card"),
                "transaction_id": str(uuid4()),
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error confirming payment: {e}")
            raise
    
    async def refund_payment(
        self,
        payment_intent_id: str,
        amount: Optional[float] = None,
        reason: str = "requested_by_customer"
    ) -> Dict[str, Any]:
        """Process a refund."""
        try:
            # Mock refund processing
            return {
                "refund_id": str(uuid4()),
                "payment_intent_id": payment_intent_id,
                "amount": amount,
                "status": "succeeded",
                "reason": reason,
                "processed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing refund: {e}")
            raise


class RevenueSharing:
    """Revenue sharing system for creators."""
    
    def __init__(self):
        self.default_creator_share = 0.70  # 70% to creator, 30% platform fee
        self.tier_multipliers = {
            "bronze": 1.0,
            "silver": 1.05,
            "gold": 1.10,
            "platinum": 1.15
        }
    
    async def calculate_revenue_split(
        self,
        gross_revenue: float,
        creator_id: UUID,
        item_type: str,
        creator_tier: str = "bronze"
    ) -> Dict[str, float]:
        """Calculate revenue split between creator and platform."""
        try:
            base_creator_share = self.default_creator_share
            
            # Adjust based on item type
            if item_type == "plugin":
                base_creator_share = 0.75  # Higher share for plugins
            elif item_type == "template":
                base_creator_share = 0.70
            elif item_type == "theme":
                base_creator_share = 0.65
            
            # Apply tier multiplier
            tier_multiplier = self.tier_multipliers.get(creator_tier, 1.0)
            final_creator_share = min(0.85, base_creator_share * tier_multiplier)  # Cap at 85%
            
            creator_revenue = gross_revenue * final_creator_share
            platform_revenue = gross_revenue - creator_revenue
            
            return {
                "gross_revenue": gross_revenue,
                "creator_revenue": creator_revenue,
                "platform_revenue": platform_revenue,
                "creator_share_percent": final_creator_share * 100,
                "tier_multiplier": tier_multiplier
            }
            
        except Exception as e:
            logger.error(f"Error calculating revenue split: {e}")
            raise


class MonetizationService:
    """Comprehensive monetization service."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.pricing_engine = PricingEngine()
        self.payment_processor = PaymentProcessor()
        self.revenue_sharing = RevenueSharing()
    
    async def configure_item_monetization(
        self,
        item_id: UUID,
        item_type: str,
        owner_id: UUID,
        monetization_config: Dict[str, Any]
    ) -> UUID:
        """Configure monetization settings for an item."""
        try:
            # Validate configuration
            await self._validate_monetization_config(monetization_config)
            
            # Create monetization configuration
            config_id = uuid4()
            
            # In a real implementation, this would create database records
            logger.info(f"Configured monetization for {item_type} {item_id}")
            
            return config_id
            
        except Exception as e:
            logger.error(f"Error configuring monetization: {e}")
            raise
    
    async def get_item_pricing(
        self,
        item_id: UUID,
        item_type: str,
        user_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get pricing information for an item."""
        try:
            # Get item data (mock)
            item_data = {
                "pricing_model": "freemium",
                "base_price": 9.99,
                "free_usage_limit": 10,
                "premium_price": 19.99,
                "currency": "USD"
            }
            
            # Get user data if provided
            user_data = None
            if user_id:
                user_data = await self._get_user_data(user_id)
            
            # Calculate pricing
            pricing_info = await self.pricing_engine.calculate_price(
                item_data, user_data
            )
            
            return pricing_info
            
        except Exception as e:
            logger.error(f"Error getting item pricing: {e}")
            raise
    
    async def initiate_purchase(
        self,
        buyer_id: UUID,
        item_id: UUID,
        item_type: str,
        payment_method: str = "card",
        coupon_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initiate a purchase transaction."""
        try:
            # Get pricing
            pricing_info = await self.get_item_pricing(item_id, item_type, buyer_id)
            
            # Apply coupon if provided
            if coupon_code:
                pricing_info = await self._apply_coupon(pricing_info, coupon_code)
            
            # Create payment intent
            payment_intent = await self.payment_processor.create_payment_intent(
                amount=pricing_info["price"],
                currency=pricing_info["currency"],
                customer_id=buyer_id,
                item_id=item_id,
                payment_method=payment_method
            )
            
            # Create purchase record
            purchase_id = uuid4()
            
            return {
                "purchase_id": str(purchase_id),
                "payment_intent": payment_intent,
                "pricing_info": pricing_info,
                "status": "pending_payment"
            }
            
        except Exception as e:
            logger.error(f"Error initiating purchase: {e}")
            raise
    
    async def complete_purchase(
        self,
        purchase_id: UUID,
        payment_intent_id: str,
        payment_method_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Complete a purchase after successful payment."""
        try:
            # Confirm payment
            payment_result = await self.payment_processor.confirm_payment(
                payment_intent_id, payment_method_data
            )
            
            if payment_result["status"] != "succeeded":
                raise ValueError("Payment was not successful")
            
            # Update purchase record
            # In real implementation, would update database
            
            # Grant access to item
            await self._grant_item_access(purchase_id)
            
            # Process revenue sharing
            await self._process_revenue_sharing(purchase_id, payment_result)
            
            return {
                "purchase_id": str(purchase_id),
                "status": "completed",
                "payment_result": payment_result,
                "completed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error completing purchase: {e}")
            raise
    
    async def create_subscription(
        self,
        user_id: UUID,
        plan_id: str,
        billing_cycle: str = "monthly"
    ) -> Dict[str, Any]:
        """Create a subscription for premium features."""
        try:
            subscription_id = uuid4()
            
            # Get plan details
            plan_details = await self._get_subscription_plan(plan_id)
            
            # Create subscription
            subscription = {
                "id": str(subscription_id),
                "user_id": str(user_id),
                "plan_id": plan_id,
                "status": "active",
                "billing_cycle": billing_cycle,
                "price": plan_details["price"],
                "currency": plan_details["currency"],
                "features": plan_details["features"],
                "created_at": datetime.utcnow().isoformat(),
                "next_billing_date": (
                    datetime.utcnow() + 
                    timedelta(days=30 if billing_cycle == "monthly" else 365)
                ).isoformat()
            }
            
            return subscription
            
        except Exception as e:
            logger.error(f"Error creating subscription: {e}")
            raise
    
    async def get_user_purchases(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get user's purchase history."""
        try:
            # Mock purchase data
            purchases = [
                {
                    "purchase_id": str(uuid4()),
                    "item_id": str(uuid4()),
                    "item_type": "plugin",
                    "item_name": "Advanced Code Formatter",
                    "price": 9.99,
                    "currency": "USD",
                    "status": "completed",
                    "purchased_at": "2024-01-15T10:30:00Z"
                }
            ]
            
            return purchases
            
        except Exception as e:
            logger.error(f"Error getting user purchases: {e}")
            return []
    
    async def get_creator_analytics(self, creator_id: UUID) -> Dict[str, Any]:
        """Get revenue and sales analytics for a creator."""
        try:
            # Mock analytics data
            analytics = {
                "total_revenue": 1250.75,
                "monthly_revenue": 324.50,
                "total_sales": 156,
                "monthly_sales": 23,
                "top_selling_items": [
                    {
                        "item_id": str(uuid4()),
                        "name": "React Component Library",
                        "sales": 45,
                        "revenue": 675.50
                    }
                ],
                "revenue_by_month": [
                    {"month": "2024-01", "revenue": 234.25},
                    {"month": "2024-02", "revenue": 345.75},
                    {"month": "2024-03", "revenue": 324.50}
                ]
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting creator analytics: {e}")
            return {}
    
    async def _validate_monetization_config(self, config: Dict[str, Any]) -> None:
        """Validate monetization configuration."""
        required_fields = ["pricing_model"]
        
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        pricing_model = config["pricing_model"]
        if pricing_model not in ["free", "one_time", "subscription", "usage_based", "tiered", "freemium"]:
            raise ValueError(f"Invalid pricing model: {pricing_model}")
    
    async def _get_user_data(self, user_id: UUID) -> Dict[str, Any]:
        """Get user data for pricing calculations."""
        # Mock user data
        return {
            "is_student": False,
            "first_time_buyer": False,
            "purchase_count": 3,
            "subscription_tier": "silver"
        }
    
    async def _apply_coupon(self, pricing_info: Dict[str, Any], coupon_code: str) -> Dict[str, Any]:
        """Apply coupon discount to pricing."""
        # Mock coupon application
        if coupon_code == "SAVE20":
            original_price = pricing_info["price"]
            discount = original_price * 0.20
            pricing_info.update({
                "original_price": original_price,
                "price": original_price - discount,
                "discount_amount": discount,
                "coupon_code": coupon_code,
                "discount_reason": "Coupon: SAVE20"
            })
        
        return pricing_info
    
    async def _grant_item_access(self, purchase_id: UUID) -> None:
        """Grant access to purchased item."""
        logger.info(f"Granted access for purchase {purchase_id}")
    
    async def _process_revenue_sharing(self, purchase_id: UUID, payment_result: Dict[str, Any]) -> None:
        """Process revenue sharing with creator."""
        logger.info(f"Processed revenue sharing for purchase {purchase_id}")
    
    async def _get_subscription_plan(self, plan_id: str) -> Dict[str, Any]:
        """Get subscription plan details."""
        plans = {
            "basic": {
                "price": 9.99,
                "currency": "USD",
                "features": ["basic_templates", "plugin_installs"]
            },
            "pro": {
                "price": 19.99,
                "currency": "USD", 
                "features": ["premium_templates", "unlimited_plugins", "priority_support"]
            },
            "enterprise": {
                "price": 49.99,
                "currency": "USD",
                "features": ["all_features", "custom_integrations", "dedicated_support"]
            }
        }
        
        return plans.get(plan_id, plans["basic"])