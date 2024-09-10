import base64
from playwright.async_api import async_playwright
import asyncio
import time

with open("mark_page.js") as f:
    mark_page_script = f.read()

async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception:
            print("Exception")
    screenshot = await page.screenshot(full_page=True)
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }


async def AsyncWebpageBroswer(url):
    browser = await async_playwright().start()
    # We will set headless=False so we can watch the agent navigate the web.
    browser = await browser.chromium.launch(headless=False, args=None)
    page = await browser.new_page()
    _ = await page.goto(url)
    return page

def save_base64_image(data, filename):
  """Saves a base64 encoded image to disk.

  Args:
      data: The base64 encoded image data (string).
      filename: The filename to save the image as (string).
  """

  try:
      # Remove any potential data URL prefix (e.g., 'data:image/png;base64,')
      img_data = base64.b64decode(data.split(",", 1)[1])

      with open(filename, "wb") as f:
          f.write(img_data)

      print(f"Image saved successfully as {filename}")
  except Exception as e:
      print(f"Error saving image: {e}")

async def get_annotated_webpage(url):
    page = await AsyncWebpageBroswer(url)
    result = await mark_page(page)
    base64_image = result['img']
    image_data = base64.b64decode(base64_image)
    with open("webpageScreenshot.png", "wb") as fh:
        fh.write(image_data)
    return f"data:image/png;base64,{base64_image}"

asyncio.run(get_annotated_webpage("https://en.wikipedia.org/wiki/Main_Page"))
