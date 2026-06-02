// This script listens for when the user clicks the extension's icon.
chrome.action.onClicked.addListener((tab) => {
  // When the icon is clicked, it runs the content.js script on the current webpage.
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content.js']
  });
});// This script listens for when the user clicks the extension's icon.
chrome.action.onClicked.addListener((tab) => {
  // When the icon is clicked, it runs the content.js script on the current webpage.
  chrome.scripting.executeScript({
    target: { tabId: tab.id },
    files: ['content.js']
  });
});