<!-- #include virtual="/counter/counter.asp" -->
<%
kwoot=""
vraagje=kwoot & request("vraagje") & kwoot

%>
<head>
<title>V r a a g j e ?</title>

<SCRIPT LANGUAGE=JAVASCRIPT>
<!--
    function subForms() {
    atw.submit();
    google.submit();
    vivi.submit();
    return false;
    }
    //--></SCRIPT>
</head>

<body>
<center>
<h1>V r a a g j e ? </h1>
<table border=1>
<tr>
<td>
U zoekt 
<%if vraagje="" then 
	response.write "<b>Nog niets</b> "
else
	response.write "<b>" & vraagje & "</b>. Klik op `Stap 2´ om dit met drie zoekmachines te zoeken."
end if
%>

</td>
</tr>

<tr>

<td>
	<form action="http://www.yepcheck.com/vraagje.asp" method="post" >
	<input name="vraagje" size="35" value="<%=vraagje%>" onClick="this.value=''">
	<input type="submit" value="Stap 1" >

<%if vraagje="" then
	'do noting
else
%>
	<INPUT TYPE="BUTTON" VALUE="Stap 2" onClick="subForms()">
	

</td>

</tr>


</FORM>

<%end if%>

<form name="atw" action="http://www.alltheweb.com/search" method="get" target="_blank">
	<input name="q"  type="hidden" value="<%=vraagje%>" >
	<input name="cat" type="hidden" value="web" /> 
 	<input name="type" value="phrase" type="hidden">
	<input name="charset" type="hidden" value="selected" />
</form>
<FORM name="google"  method=GET action="http://www.google.nl/search"  target="_blank">
	<INPUT type="hidden"  name=as_epq  value="<%=vraagje%>">
	<INPUT TYPE=hidden name=hl value="nl">
</FORM>
<FORM name="vivi" action="http://vivisimo.com/search"  target="_blank">
	<INPUT type="hidden" name="v:sources" value="EnglishWeb">
	<INPUT type=hidden name="query"  value="<%=vraagje%>">
</form>

</TABLE>
<h2>Hoe werkt het</h2>
<ul>
<li>Je tikt iets in het tekstvak
<li>Je klikt op `stap 1´.</li>
<li>Je klikt op `stap 2´. (button verschijnt zodra je op `stap 1´ hebt geklikt.)</li>
<li>Er worden nu drie zoekmachinevensters geopend</li>
<li>Door op het tekstvak te klikken verdwijnt de tekst en begin je helemaal opnieuw</li>
</ul>

<h2>Wishlist</h2>
<ul>
<li>Nu moet je nog twee keer klikken. Straks niet meer</li>
<li>Nu moet de popupblockker van Google nog uit staan. Straks niet meer</li>
<li>Ik heb nu alles op "exact prase" (omdat ik vaak naar namen van personen zoek. Dit wordt natuurlijk "en", "of", "excact phrase etc"</li>
<li>Mooie doch simpele opmaak</li>
<li>Een business Angle die er een boel geld instopt... Oeps... drie jaar te laat!</li>
</ul>
<h2>Andere slimme tools van René Smit</h2>
<ul>
<li><A HREF="http://www.yepcheck.com/name" target="_blank">Naam-generator</a></li>
<li><A HREF="http://pianotab.yepcheck.com" target="_blank">Gitaar-naar-piantab-convertor</a></li>
<li><A HREF="http://www.yepcheck.com" target="_blank">A yepcheck can make the difference!</a></li>
</ul>
<h2><A HREF="http://www.yepcheck.com/contact.asp" target="_blank">Rene mailen</a></h2>

<P>
(Advertentie)<br>
<A HREF="http://www.peeze.nl/barista" target="_blank"><IMG SRC="http://www.peeze.nl/barista/popuplogo.gif"  alt="logo van de Dutch Barista Championship" border="0"></a>
<FONT COLOR="#FFFFFF">
<h2>
De website over de Baristakampioenschappen 2004 is nu online. Klik op het logo om deze te bekijken</h2>
</p>
</center>
</body>

