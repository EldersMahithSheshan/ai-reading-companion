// 1. Check if the tooltip already exists to prevent the "Identifier declared" error
let tooltip = document.getElementById('ai-reading-tooltip-unique-id');

if (!tooltip) {
    tooltip = document.createElement('div');
    tooltip.id = 'ai-reading-tooltip-unique-id'; // Give it an ID so we can find it later
    tooltip.className = 'ai-reading-tooltip';
    document.body.appendChild(tooltip);
}

// 2. Highlight Logic (Calls /analyze) - UPDATED FOR STABILITY
async function highlightComplexWords() {
    const existingHighlights = document.querySelectorAll('.cwi-highlight');
    if (existingHighlights.length > 0) {
        existingHighlights.forEach(span => {
            const text = document.createTextNode(span.innerText);
            span.parentNode.replaceChild(text, span);
        });
        document.body.normalize();
        return;
    }

    const paragraphs = document.querySelectorAll('p');
    let fullText = "";
    paragraphs.forEach(p => { fullText += p.innerText + " "; });

    if (fullText.trim().length === 0) return;

    try {
        const response = await fetch('http://127.0.0.1:5000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: fullText })
        });

        const data = await response.json();
        const complexWords = data.complex_words;

        if (!complexWords || complexWords.length === 0) return;

        complexWords.sort((a, b) => b.length - a.length);
        const regex = new RegExp(`\\b(${complexWords.join('|')})\\b`, 'gi');

        // SAFE REPLACEMENT STRATEGY:
        // Instead of p.innerHTML.replace (which breaks tags), we process text nodes only.
        paragraphs.forEach(p => {
            const walker = document.createTreeWalker(p, NodeFilter.SHOW_TEXT, null, false);
            let node;
            const nodesToReplace = [];

            while (node = walker.nextNode()) {
                if (regex.test(node.nodeValue)) {
                    nodesToReplace.push(node);
                }
            }

            nodesToReplace.forEach(textNode => {
                const parent = textNode.parentNode;
                // Skip if we are already inside a highlight or a link that might have attributes
                if (parent.className === 'cwi-highlight' || parent.tagName === 'A') return;

                const html = textNode.nodeValue.replace(regex, match => {
                    return `<span class="cwi-highlight" data-word="${match}">${match}</span>`;
                });

                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = html;
                
                while (tempDiv.firstChild) {
                    parent.insertBefore(tempDiv.firstChild, textNode);
                }
                parent.removeChild(textNode);
            });
        });

        attachHoverEvents();

    } catch (e) {
        console.error("CWI Backend Error:", e);
    }
}

// 3. Definition Logic (Calls /define)
function attachHoverEvents() {
    // We re-select highlights because they are new elements now
    const highlights = document.querySelectorAll('.cwi-highlight');
    
    highlights.forEach(span => {
        span.addEventListener('mouseenter', async (e) => {
            // UPDATE: Get the word from the data attribute, NOT innerText.
            // This ensures we don't accidentally grab punctuation or spaces.
            const word = e.target.getAttribute('data-word');
            
            // Grab context from the paragraph
            const context = e.target.closest('p') ? e.target.closest('p').innerText.substring(0, 150) + "..." : "General context";

            // Position Tooltip
            const rect = e.target.getBoundingClientRect();
            tooltip.style.display = 'block';
            tooltip.style.left = (window.scrollX + rect.left) + 'px';
            tooltip.style.top = (window.scrollY + rect.bottom + 5) + 'px';
            
            tooltip.innerHTML = `<strong>${word}</strong><br><em>Loading Sinhala definition...</em>`;

            try {
                const response = await fetch('http://127.0.0.1:5000/define', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ word: word, context: context })
                });
                
                const data = await response.json();
                tooltip.innerHTML = `<strong>${word}</strong><br>${data.definition}`;
            } catch (err) {
                tooltip.innerHTML = `<strong>${word}</strong><br>Error loading definition.`;
                console.error(err);
            }
        });

        span.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
    });
}

// Run the function
highlightComplexWords();