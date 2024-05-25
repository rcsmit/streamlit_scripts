<!-- #include virtual="/counter/counter.asp" -->
<%
    '**************************************
    ' Name: List all files in a directory
    ' Description:Uses the new file FileSyst
    '     emObject in the scripting library to lis
    '     t all the files in the c:\inetpub\script
    '     s\ directory with a link to them. You ca
    '     n modify this code to list all the files
    '     in any directory.
    ' By: Ian Ippolito (RAC)
    '
    'This code is copyrighted and has    
    'limited warranties.Please see http://w
    'ww.Planet-Source-Code.com/vb/scripts/ShowCode.asp?txtCodeId=44&lngWId=4    
    'for details.    
    '**************************************
    %>
    <HTML>
<head>
<TITLE>Directorylisting</title>
<base target="_blank">
</head>

    <BODY>

<table>
    <%
    Dim objFileScripting, objFolder
    Dim filename, filecollection, strDirectoryPath, strUrlPath, size, showdatecreated,f,s
    	'strDirectoryPath="d:\www\yepcheck.com\www\printbak\"
  	
    	'get file scripting object
    	Set objFileScripting = CreateObject("Scripting.FileSystemObject")
    	'Return folder object

    	'Set objFolder = objFileScripting.GetFolder("d:\www\yepcheck.com\www\printbak\")

	foldername=objFileScripting.GetFolder(server.mappath(".")).name
	Set objFolder = foldername




' set fso = createobject("scripting.filesystemobject")
'foldername = fso.getfolder(server.mappath(".")).name
'response.write foldername 





    	'return file collection In folder
    	Set filecollection = objFolder.Files
    	'create the links
  	For Each filename In filecollection
		response.write "<tr><td>"
		Set f = objFileScripting.GetFile(filename)
		s = s & "<A HREF='" &  f.name & "'>" & f.name & "</A><BR>"
		if right(filename,4)= ".jpg" or right(filename,4)= ".gif" OR right(filename,4)= ".tif" OR right(filename,4)= ".JPG" or right(filename,4)= ".GIF" OR right(filename,4)= ".TIF" then
			s=s&"<A HREF='" &  f.name & "'><IMG SRC='"&f.name&"' width='400' border='0'></a><br>"
		end if 


   s = s & "Created: " & f.DateCreated & "<BR>"
   s = s & "Last Accessed: " & f.DateLastAccessed & "<BR>"
   s = s & "Last Modified: " & f.DateLastModified   & "<BR>"
   s = s & "size: " & f.size& " bits<BR>"
   s = s&"<HR>"
response.write s
s=""

response.write "</td></tr>"




    	Next
    %>
    </BODY>
    </HTML>


