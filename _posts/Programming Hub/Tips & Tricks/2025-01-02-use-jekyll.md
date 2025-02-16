---
layout: post
title: Use Jekyll
date: 2025-01-02 09:00 -0800
description: Examples of using Jekyll
authors: [cotes, khoa_pham]
categories: [Programming Hub, Tips & Tricks]
tags: [blogging, settings]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

> Link to [Chirpy](https://chirpy.cotes.page)  
> [Jekyll & Ruby Setups](https://jekyllrb.com/docs/installation/)  

<!-- --- -->

## Running Locally

> Links to test locally:  
> <http://127.0.0.1:4000/>  
> <http://127.0.0.1:4000/mindpalace/>  

```bash
bundle install
```

```bash
bundle exec jekyll serve
```

## Managing Posts

> Link to [Jekyll Compose](https://github.com/jekyll/jekyll-compose)

### Create a Post

```bash
bundle exec jekyll post "My New Post"
```

```bash
bundle exec jekyll compose "My New Post"
```

```bash
bundle exec jekyll post "My New Post" --timestamp-format "%Y-%m-%d %H:%M:%S %z"  
bundle exec jekyll compose "My New Post 1" --post
bundle exec jekyll compose "My New Post 2" --collection "posts"
```

### Rename a Post

```bash
bundle exec jekyll rename _posts/2014-01-24-my-new-draft.md "My New Post"
```

```bash
bundle exec jekyll rename _posts/2012-03-04-my-old-post.md "My New Post" --now
bundle exec jekyll rename _posts/2014-01-24-my-new-post.md "My Old Post" --date "2012-03-04"
```

## Front Matter

```yaml

---
layout: post
title: TITLE
date: YYYY-MM-DD HH:MM:SS +/-TTTT
description: Short summary of the post.
categories: [Top_Categ, Sub_Categ]
tags: [tag1, tag2, tag3, tag4]
authors: [<author1_id>, <author2_id>]
toc: true
comments: false
pin: true
math: true
mermaid: true

<author_id>:
  name: <full name>
  twitter: <twitter_of_author>
  url: <homepage_of_author>

image:
  path: /path/to/image
  alt: image with 1200 x 630 or "1.91 to 1" aspect ratio
---

```

## Use Markdown 
> Link to [Use Markdown](https://khoapham1002.github.io/mindpalace/posts/use-markdown/)  