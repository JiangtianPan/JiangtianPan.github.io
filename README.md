# jiangtianpan.github.io

Personal site of Yukin (Jiangtian) Pan — computer vision researcher.
Live at **<https://jiangtianpan.github.io>**.

Static HTML, CSS and JavaScript. No framework, no build step, no dependencies —
push to `main` and GitHub Pages serves it.

## Structure

```
index.html            Home — bio, updates, project links
resume.html           Full résumé
demo.html             Project demos (video processing + web)
css/site.css          The entire stylesheet: design tokens, layout, components
js/site.js            Theme toggle + footer year
images/avatar.jpg     Profile image
documents/
  Resume_JiangtianPan.pdf    Résumé, PDF
  Resume_YukinPan.md         Résumé, Markdown source
  demo/                      Demo videos and stills
  demo/posters/              Video poster frames (generated, see below)
```

## Working on it

Open any page directly in a browser, or serve the folder to get correct
relative paths everywhere:

```sh
python3 -m http.server 8000
```

### Editing content

All content lives in the three HTML files as plain markup. The shared header
and footer are duplicated across them by design — with three pages that is
cheaper than adopting a template engine. If you add a fourth page, copy an
existing one and update the `aria-current="page"` marker on its nav link.

### Editing style

Everything is driven by the custom properties at the top of `css/site.css`.
Changing `--c-accent` or `--w-prose` there restyles the whole site; the rest of
the file is layout and components built on those tokens.

Light and dark palettes are both defined. Dark follows the OS by default and
the header toggle overrides it, persisting the choice in `localStorage`.

### Regenerating video posters

Demo videos use `preload="none"` with a poster frame, so the ~180 MB of MP4 is
only fetched when a visitor presses play. After adding or replacing a video,
regenerate its poster:

```sh
ffmpeg -ss 00:00:01 -i documents/demo/NAME.mp4 -frames:v 1 \
       -vf "scale=720:-2" -q:v 4 documents/demo/posters/NAME.jpg
```

## Résumé

The résumé exists in three places, kept in sync by hand:
`resume.html` (canonical, on the web), `documents/Resume_JiangtianPan.pdf`
(the download), and `documents/Resume_YukinPan.md` (Markdown source).
