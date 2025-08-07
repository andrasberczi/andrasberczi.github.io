# Andras Berczi's Blog

Personal blog built with Jekyll and the beautiful-jekyll-theme.

## Local Development

### Prerequisites

- Ruby 3.4.1 (managed via rbenv)
- Bundler

### Setup

1. **Install dependencies:**

   ```bash
   bundle install
   ```

2. **Start the local server:**

   ```bash
   eval "$(rbenv init -)" && bundle exec jekyll serve --livereload
   ```

3. **Visit your site:**
   - Main site: http://127.0.0.1:4000/
   - Blog post: http://127.0.0.1:4000/2025/08/07/what-mean-can-mean.html

### Troubleshooting

**Port conflicts:** If you get "port is in use" errors, try:

```bash
eval "$(rbenv init -)" && bundle exec jekyll serve --livereload --port 4001
```

**Ruby version issues:** Make sure you're using Ruby 3.4.1:

```bash
rbenv local 3.4.1
eval "$(rbenv init -)"
ruby --version  # Should show 3.4.1
```

### Features

- ✅ LaTeX rendering with MathJax
- ✅ Beautiful Jekyll theme
- ✅ Live reload for development
- ✅ GitHub Pages compatible

### Adding LaTeX to Posts

For posts that need LaTeX rendering, add this at the very end of the post content:

```html
<!-- MathJax Configuration -->
<script>
  window.MathJax = {
    tex: {
      inlineMath: [
        ["$", "$"],
        ["\\(", "\\)"],
      ],
      displayMath: [
        ["$$", "$$"],
        ["\\[", "\\]"],
      ],
      processEscapes: true,
      processEnvironments: true,
    },
    options: {
      ignoreHtmlClass: "tex2jax_ignore",
      processHtmlClass: "tex2jax_process",
    },
  };
</script>
<script
  type="text/javascript"
  id="MathJax-script"
  async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
></script>

<style>
  .MathJax {
    font-size: 1.1em;
  }
  .MathJax_Display {
    margin: 1em 0;
  }
</style>
```

**Important:** Place this at the very end of the post to avoid breaking the excerpt preview.
