import { test, expect } from '@playwright/test';

test('has title', async ({ page }) => {
  await page.goto('http://localhost:3000');

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/Mosaic/);
});

test('login page renders', async ({ page }) => {
  await page.goto('http://localhost:3000/login');

  // Expect the page to have a login header or button
  await expect(page.locator('text=Sign in').first()).toBeVisible();
});
