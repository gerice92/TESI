function httpGet(theUrl)
{
	var server = "http://localhost:8080/hello?url=";
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", server + theUrl, false ); // false for synchronous request
    xmlHttp.send();
    return xmlHttp.responseText;
}

function sendRequest()
{
	//document.getElementById("status").innerHTML = "loading...";
	chrome.tabs.query({'active': true, 'lastFocusedWindow': true}, function (tabs) {
    	var url = tabs[0].url;
    	var response = httpGet(url);
 		var newWindow = window.open();
		newWindow.document.write(response);
	});
}


//window.onload = httpGet(window.location.href);
window.onload = sendRequest();