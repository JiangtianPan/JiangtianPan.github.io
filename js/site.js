/**
 * Site behaviour: theme toggle + footer year.
 *
 * The stored theme is applied by a tiny inline script in <head> so the page
 * never paints with the wrong palette. This file only handles interaction.
 *
 * Theme state has three values, not two: "light", "dark", and *unset* —
 * unset means follow the OS, and CSS handles that on its own via
 * prefers-color-scheme. Nothing here writes storage until the visitor
 * actually clicks the toggle, so following the OS survives a page load.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "theme";
  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  var toggle = document.querySelector(".theme-toggle");

  function read() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (e) {
      return null; /* private mode / storage disabled */
    }
  }

  function write(value) {
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch (e) {
      /* theme just won't persist across loads */
    }
  }

  /** The theme actually on screen right now. */
  function resolvedTheme() {
    var explicit = root.getAttribute("data-theme");
    if (explicit === "dark" || explicit === "light") return explicit;
    return media.matches ? "dark" : "light";
  }

  /** Keep the mobile browser chrome and the button label in step. */
  function syncUi(theme) {
    var tag = document.querySelector('meta[name="theme-color"]');
    if (tag) {
      tag.setAttribute("content", theme === "dark" ? "#14151a" : "#fdfdfc");
    }
    if (toggle) {
      var label = "Switch to " + (theme === "dark" ? "light" : "dark") + " theme";
      toggle.setAttribute("aria-label", label);
      toggle.setAttribute("title", label);
    }
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = resolvedTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      write(next);
      syncUi(next);
    });
  }

  // Follow the OS only while the visitor has not made an explicit choice.
  media.addEventListener("change", function () {
    if (!read()) syncUi(media.matches ? "dark" : "light");
  });

  syncUi(resolvedTheme());

  var year = document.querySelector("[data-year]");
  if (year) year.textContent = String(new Date().getFullYear());
})();
