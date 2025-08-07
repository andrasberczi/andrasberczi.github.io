source "https://rubygems.org"

# Specify Ruby version compatibility
ruby "~> 3.4.0"

# Jekyll and theme
gem "jekyll", "~> 3.9.0"
gem "beautiful-jekyll-theme", "5.0.0"

# GitHub Pages compatibility
gem "github-pages", "~> 228", group: :jekyll_plugins

# Plugins
group :jekyll_plugins do
  gem "jekyll-feed", "~> 0.15"
end

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Webrick for Ruby 3.0+
gem "webrick", "~> 1.8"
