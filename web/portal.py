import http.server
import socketserver

class PortalHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = 'index.html'  
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

# Create an object of the above class
handler_object = PortalHttpRequestHandler

PORT = 80
my_server = socketserver.TCPServer(("", PORT), handler_object)

# Star the server
my_server.serve_forever()