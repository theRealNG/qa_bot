from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
import base64
import asyncio


async def AsyncWebpageScreenshot(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        stealth_async(page)
        await page.goto(url)

        screenshot_bytes = await page.screenshot(full_page=True)

        await browser.close()

    base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
    return base64_image


def WebpageScreenshot(url):
    print("Taking screenshot: ", url)
    result = asyncio.run(AsyncWebpageScreenshot(url))
    return result
