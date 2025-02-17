---
layout: post
title: Use Markdown
date: 2025-01-02 10:00 -0800
description: Examples of using Markdown
authors: [cotes, khoa_pham]
categories: [Programming Hub, Tips & Tricks]
tags: [blogging, settings]
pin: false
math: true
mermaid: true
toc: true
comments: true
---

[Writing a New Post](https://chirpy.cotes.page/posts/write-a-new-post/)[^fn1]   
> <https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/_posts/2019-08-08-write-a-new-post.md?plain=1>

[Text and Typography](https://chirpy.cotes.page/posts/text-and-typography/)[^fn2]   
> <https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/_posts/2019-08-08-text-and-typography.md?plain=1>

<br>
Link to [Use Jekyll](https://khoapham1002.github.io/mindpalace/posts/use-jekyll/)   
Jekyll is a static site generator that transforms plain text into static websites and blogs. When used together with Markdown, Jekyll allows users to write content in a simple, readable format that can be easily converted into HTML. Markdown's straightforward syntax makes it easy to format text, create lists, add links, and include code snippets, while Jekyll handles the layout and structure of the site. This combination provides a powerful yet user-friendly way to create and manage static websites.

## Headings

```markdown
# Header 1
## Header 2
### Header 3
#### Header 4
##### Header 5
###### Header 6
```

## Emphasis

*italic* or _italic_ <br>
**bold** or __bold__ <br>
***bold and italic*** or ___bold and italic___ <br>
This is <strong>bold</strong> text using HTML syntax. <br>

~~This is strikethrough text.~~

```markdown
*italic* or _italic_ <br>
**bold** or __bold__ <br>
***bold and italic*** or ___bold and italic___ <br>
This is <strong>bold</strong> text using HTML syntax. <br>

~~This is strikethrough text.~~
```

## Ordered list

1. Firstly
2. Secondly
3. Thirdly

```markdown
1. Firstly
2. Secondly
3. Thirdly
```

## Unordered list

- Chapter
  - Section
    - Paragraph

```markdown
- Chapter
  - Section
    - Paragraph
```

## ToDo list

- [ ] Job
  - [x] Step 1
  - [x] Step 2
  - [ ] Step 3

```markdown
- [ ] Job
  - [x] Step 1
  - [x] Step 2
  - [ ] Step 3
```

## Description list

***Term 1***
: ~~Definition of term 1~~

**Term 2**
: _Definition of term 2_

```markdown
***Term 1***
: ~~Definition of term 1~~

**Term 2**
: _Definition of term 2_
```

## Tables

| Header 1 | Header 2 |
| -------- | -------- |
| Row 1 Col 1 | Row 1 Col 2 |
| Row 2 Col 1 | Row 2 Col 2 |

```markdown
| Header 1 | Header 2 |
| -------- | -------- |
| Row 1 Col 1 | Row 1 Col 2 |
| Row 2 Col 1 | Row 2 Col 2 |
```

| Left Aligned | Center Aligned | Right Aligned |
|:------------ |:--------------:| -------------:|
| Row 1 Col 1  | Row 1 Col 2    | Row 1 Col 3   |
| Row 2 Col 1  | Row 2 Col 2    | Row 2 Col 3   |

```markdown
| Left Aligned | Center Aligned | Right Aligned |
|:------------ |:--------------:| -------------:|
| Row 1 Col 1  | Row 1 Col 2    | Row 1 Col 3   |
| Row 2 Col 1  | Row 2 Col 2    | Row 2 Col 3   |
```

## Block Quote

> This line shows the _block quote_.  
>> This line shows the _nested block quote_.  
>>> This line shows the _nested block quote_.

>>>> This line shows the _nested block quote_.  
>> This line shows the _nested block quote_.  

```markdown
> This line shows the _block quote_.  
>> This line shows the _nested block quote_.  
>>> This line shows the _nested block quote_.

>>>> This line shows the _nested block quote_.  
>> This line shows the _nested block quote_.  
```

## Prompts

> An example showing the `tip` type prompt.
{: .prompt-tip }

> An example showing the `info` type prompt.
{: .prompt-info }

> An example showing the `warning` type prompt.
{: .prompt-warning }

> An example showing the `danger` type prompt.
{: .prompt-danger }

```markdown
> An example showing the `tip` type prompt.
{: .prompt-tip }

> An example showing the `info` type prompt.
{: .prompt-info }

> An example showing the `warning` type prompt.
{: .prompt-warning }

> An example showing the `danger` type prompt.
{: .prompt-danger }
```

## Inline Code

This is an example of `Inline Code`.

```markdown
This is an example of `Inline Code`.
```

## Code blocks

Using ```` ```{language} ```` you will get a code block with syntax highlight:

````markdown
{% raw %}```python
x = 10
if x > 5:
    print("x is greater than 5")
else:
    print("x is not greater than 5") 
```{% endraw %}
````

## Math & LaTeX

The mathematics powered by [**MathJax**](https://www.mathjax.org/):

When $$a \ne 0$$, there are two solutions to $$ax^2 + bx + c = 0$$ and they are:    
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

$$ x = \frac{-b \pm \sqrt{b^2-4ac}}{2a} $$

```markdown
When $$a \ne 0$$, there are two solutions to $$ax^2 + bx + c = 0$$ and they are:    
$$ x = {-b \pm \sqrt{b^2-4ac} \over 2a} $$

$$ x = \frac{-b \pm \sqrt{b^2-4ac}}{2a} $$

```

1. For markdown, LaTex inline equation: $$E = m \times c^2$$
2. For HTML rendering inline equation: \\( E = mc^2 \\)
3. \$$E = mc^2$$

```markdown
1. For markdown, LaTex inline equation: $$E = m \times c^2$$
2. For HTML rendering inline equation: \\( E = mc^2 \\)
3. \$$E = mc^2$$
```

This is a displayed equation:
$$
a^2 + b^2 = c^2
$$

```markdown
This is a displayed equation:
$$
a^2 + b^2 = c^2
$$
```

## Mermaid SVG

```mermaid
 gantt
  title  Adding GANTT diagram functionality to mermaid
  apple :a, 2017-07-20, 1w
  banana :crit, b, 2017-07-23, 1d
  cherry :active, c, after b a, 1d
```

````markdown
{% raw %}```mermaid
 gantt
  title  Adding GANTT diagram functionality to mermaid
  apple :a, 2017-07-20, 1w
  banana :crit, b, 2017-07-23, 1d
  cherry :active, c, after b a, 1d
```{% endraw %}
````

## Liquid

````markdown
{% raw %}```liquid
{% if product.title contains 'Pack' %}
  This product's title contains the word Pack.
{% endif %}
```{% endraw %}
````

## Images

![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png)
_Potter's Doe Patronous_

![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png){: w="300" h="300" }{: .normal }{: .shadow }

<!-- ![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png){: .left }
![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png){: .right } -->

```markdown
![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png)
_Potter's Doe Patronous_

![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png){: width="700" height="400" }
![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png){: w="300" h="300" }{: .normal }{: .shadow }

![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png){: .left }
![Doe_Patronus](assets/img/favicons/web-app-manifest-512x512.png){: .right }

```

## Video

{% include embed/youtube.html id='PsP2vsy_8ms' %}

| Video URL      | Platform   | ID             |
| -------------- | ---------- | :------------- |
| [https://www.**youtube**.com/watch?v=**PsP2vsy_8ms**](https://www.youtube.com/watch?v=PsP2vsy_8ms)    | `youtube`        | `PsP2vsy_8ms`  |
| [https://www.**twitch**.tv/videos/**1634779211**](https://www.twitch.tv/videos/1634779211)            | `twitch`         | `1634779211`   |

````markdown
{% raw %}
{% include embed/{Platform}.html id='{ID}' %}
{% include embed/youtube.html id='PsP2vsy_8ms' %}

{% include embed/video.html src='{URL}' %}
Where `URL` is a URL to a video file e.g. `/path/to/sample/video.mp4`.

{%
  include embed/video.html
  src='/path/to/video.mp4'
  types='ogg|mov'
  poster='poster.png'
  title='Demo video'
  autoplay=true
  loop=true
  muted=true
%}

- `poster='/path/to/poster.png'` — poster image for a video that is shown while video is downloading
- `title='Text'` — title for a video that appears below the video and looks same as for images
- `autoplay=true` — video automatically begins to play back as soon as it can
- `loop=true` — automatically seek back to the start upon reaching the end of the video
- `muted=true` — audio will be initially silenced
- `types` — specify the extensions of additional video formats separated by `|`. Ensure these files exist in the same directory
{% endraw %}
````

## Links

Link to [Chirpy](https://chirpy.cotes.page)

```markdown
Link to [Chirpy](https://chirpy.cotes.page)
```

## Footnotes

```markdown
Click the hook will locate the footnote[^footnote], and here is another footnote[^fn-nth-2].


[^footnote]: The footnote source
[^fn-nth-2]: The 2nd footnote source
```

### Post's Footnotes

---

[^fn1]: Check `Writing a New Post` post
[^fn2]: Check `Text and Typography` post
