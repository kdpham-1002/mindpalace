---
layout: post
title: Mac Settings
date: 2025-01-03 18:30 -0800
description: MacOS shortcuts and settings
author: khoa_pham
categories: [Programming Hub, Tips & Tricks]
tags: [settings]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

## Mac Setups

### Dock Animation

[Show dock faster_StackExchange](https://apple.stackexchange.com/questions/33600/how-can-i-make-auto-hide-show-for-the-dock-faster)

```bash
# Customize
defaults write com.apple.dock autohide-delay -float 0
defaults write com.apple.dock autohide-time-modifier -float 0.7
killall Dock
```

```bash
# Reset
defaults delete com.apple.dock autohide-delay
defaults delete com.apple.dock autohide-time-modifier
killall Dock
```

### Translucent Dock Icons

[Translucent Hidden Apps in Dock_OSX Daily](https://osxdaily.com/2010/06/22/make-hidden-application-icons-translucent-in-the-dock/)

```bash
# Enable
defaults write com.apple.Dock showhidden -bool YES;
killall Dock
```

```bash
# Reset
defaults write com.apple.Dock showhidden -bool NO;
killall Dock
```

### Launchpad Grid Layout

[Change Launchpad Grid Layout_Compsmag](https://www.compsmag.com/how-to/change-launchpad-layout/)

```bash
defaults write com.apple.dock springboard-columns -int 6
defaults write com.apple.dock springboard-rows -int 5;
defaults write com.apple.dock ResetLaunchPad -bool TRUE;
killall Dock
```

### Quick Look Text Selection (optional)

```bash
# Enable
defaults write com.apple.finder QLEnableTextSelection -bool TRUE
```

## Mac Shortcuts

[Mac keyboard shortcuts](https://support.apple.com/en-us/102650)  
[30 Terminal tips & tricks for Mac](https://www.macworld.com/article/671711/30-terminal-tips-tricks-and-projects-for-mac.html)


```bash
caffeinate
```

```bash
#3600s = 1hr
caffeinate -u -t 3600
```

```bash
# View (text) files that won't open
cat [path]
```

```bash
# Show hidden files/folders
defaults write com.apple.finder AppleShowAllFiles -bool TRUE
killall Finder
# --- Cmd + Shift + . --- also works

# Hide a file/folder from view
chflags hidden [path]
```
