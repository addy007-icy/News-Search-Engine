import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from PyQt5.QtWidgets import *

"""
    Importing Query Processor
"""
import query_proc






class MainWindow(QMainWindow):
    """ The class that defines the structure of the application's GUI.
        The main GUI contains a Query Input box 2 tabs:
        1.) Results Tab : Where the output of the top 10 results would come
        2.) Sugesstions Tab : Where the output of the top 10 suggestions would come
    """
    def __init__(self, parent=None):
        super(MainWindow,self).__init__(parent)
        self.resize(1024,1024)
        self.menubar = self.menuBar()
        help_ = self.menubar.addMenu("Help")
        aboutAction = QAction("About",self)
        self.shortcut = QShortcut(QKeySequence("Alt+A"),self)
        self.shortcut.activated.connect(self.about)
        aboutAction.setToolTip("About News Search")
        aboutAction.triggered.connect(self.about)

        contributeAction = QAction("Contribute Or Report Issues",self)
        self.shortcut = QShortcut(QKeySequence("Alt+C"),self)
        self.shortcut.activated.connect(self.contribute)
        contributeAction.setToolTip("Opens Repository")
        contributeAction.triggered.connect(self.contribute)

        usageAction = QAction("Usage",self)
        self.shortcut = QShortcut(QKeySequence("Alt+U"),self)
        self.shortcut.activated.connect(self.usage)
        usageAction.setToolTip("Instructions for Using the tool")
        usageAction.triggered.connect(self.usage)

        help_.addAction(aboutAction)
        help_.addAction(contributeAction)
        help_.addAction(usageAction)

        self.dialog = QDialog()

        self.statusbar = self.statusBar()

        self.mainlayout = QVBoxLayout()
        self.grid = QGridLayout()

        font = QFont("Times",13)
        self.queryLabel = QLabel("Query:")
        self.queryLabel.setFont(font)
        self.queryInput = QLineEdit()
        self.queryInput.setFont(font)

        font = QFont("Times",13)
        self.loadLabel = QLabel("Load:")
        self.loadLabel.setFont(font)
        self.loadInput = QLineEdit()
        self.loadInput.setFont(font)

        self.button = QPushButton()
        self.button.setFont(font)
        self.button.setText("Search")
        self.button.setFixedWidth(100)
        self.button.clicked.connect(self.modifyUI)


        self.buttonP1 = QPushButton()
        self.buttonP1.setFont(font)
        self.buttonP1.setText("Load 1st result")
        self.buttonP1.setFixedWidth(1024)
        self.buttonP2 = QPushButton()
        self.buttonP2.setFont(font)
        self.buttonP2.setText("Load 2nd result")
        self.buttonP2.setFixedWidth(1024)
        self.buttonP3 = QPushButton()
        self.buttonP3.setFont(font)
        self.buttonP3.setText("Load 3th result")
        self.buttonP3.setFixedWidth(1024)
        self.buttonP4 = QPushButton()
        self.buttonP4.setFont(font)
        self.buttonP4.setText("Load 4th result")
        self.buttonP4.setFixedWidth(1024)
        self.buttonP5 = QPushButton()
        self.buttonP5.setFont(font)
        self.buttonP5.setText("Load 5th result")
        self.buttonP5.setFixedWidth(1024)
        self.buttonP6 = QPushButton()
        self.buttonP6.setFont(font)
        self.buttonP6.setText("Load 6th result")
        self.buttonP6.setFixedWidth(1024)
        self.buttonP7 = QPushButton()
        self.buttonP7.setFont(font)
        self.buttonP7.setText("Load 7th result")
        self.buttonP7.setFixedWidth(1024)
        self.buttonP8 = QPushButton()
        self.buttonP8.setFont(font)
        self.buttonP8.setText("Load 8th result")
        self.buttonP8.setFixedWidth(1024)
        self.buttonP9 = QPushButton()
        self.buttonP9.setFont(font)
        self.buttonP9.setText("Load 9th result")
        self.buttonP9.setFixedWidth(1024)
        self.buttonP10 = QPushButton()
        self.buttonP10.setFont(font)
        self.buttonP10.setText("Load 10th result")
        self.buttonP10.setFixedWidth(1024)

        self.buttonS1 = QPushButton()
        self.buttonS1.setFont(font)
        self.buttonS1.setText("Load 1st Suggestion")
        self.buttonS1.setFixedWidth(1024)
        self.buttonS2 = QPushButton()
        self.buttonS2.setFont(font)
        self.buttonS2.setText("Load 2nd Suggestion")
        self.buttonS2.setFixedWidth(1024)
        self.buttonS3 = QPushButton()
        self.buttonS3.setFont(font)
        self.buttonS3.setText("Load 3rd Suggestion")
        self.buttonS3.setFixedWidth(1024)
        self.buttonS4 = QPushButton()
        self.buttonS4.setFont(font)
        self.buttonS4.setText("Load 4th Suggestion")
        self.buttonS4.setFixedWidth(1024)
        self.buttonS5 = QPushButton()
        self.buttonS5.setFont(font)
        self.buttonS5.setText("Load 5th Suggestion")
        self.buttonS5.setFixedWidth(1024)
        self.buttonS6 = QPushButton()
        self.buttonS6.setFont(font)
        self.buttonS6.setText("Load 6rh Suggestion")
        self.buttonS6.setFixedWidth(1024)
        self.buttonS7 = QPushButton()
        self.buttonS7.setFont(font)
        self.buttonS7.setText("Load 7th Suggestion")
        self.buttonS7.setFixedWidth(1024)
        self.buttonS8 = QPushButton()
        self.buttonS8.setFont(font)
        self.buttonS8.setText("Load 8th Suggestion")
        self.buttonS8.setFixedWidth(1024)
        self.buttonS9 = QPushButton()
        self.buttonS9.setFont(font)
        self.buttonS9.setText("Load 9th Suggestion")
        self.buttonS9.setFixedWidth(1024)
        self.buttonS10 = QPushButton()
        self.buttonS10.setFont(font)
        self.buttonS10.setText("Load 10th Suggestion")
        self.buttonS10.setFixedWidth(1024)



        self.grid.addWidget(self.queryLabel,0,0)
        self.grid.addWidget(self.queryInput,0,1)
        self.grid.addWidget(self.button,0,2)

        self.mainlayout.addLayout(self.grid)

        self.tab = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab.addTab(self.tab1, "1")
        self.tab.addTab(self.tab2,"2")

            
        self.tab1layout = QVBoxLayout()
        self.ResultBrowser = QTextBrowser()
        self.ResultBrowser.append("")
        self.ResultBrowser.setFont(QFont("Courier",13))
        self.ResultBrowser.setTextColor(QColor("#C5C8C6"))
        self.ResultBrowser.setStyleSheet("background-color: #1d1f21")
        self.ResultBrowser.setText("Documents will be shown here")
        self.tab1layout.addWidget(self.ResultBrowser)
        self.tab.setTabText(0,"Result Documents")
        self.tab.setFont(font)
        self.tab1.setLayout(self.tab1layout)

        self.tab2layout = QVBoxLayout()
        self.SuggestionBrowser = QTextBrowser()
        self.SuggestionBrowser.setFont(QFont("Courier",13))
        self.SuggestionBrowser.setTextColor(QColor("#C5C8C6"))
        self.SuggestionBrowser.append("You might like: \n\n")
        self.SuggestionBrowser.setStyleSheet("background-color: #1d1f21")
        self.tab2layout.addWidget(self.SuggestionBrowser)
        self.tab.setTabText(1,"Things you might like")
        self.tab2.setLayout(self.tab2layout)


        self.mainlayout.addWidget(self.tab)
        self.dialog.setLayout(self.mainlayout)
        self.setCentralWidget(self.dialog)
        self.setWindowTitle("News Search")



    def about(self):
        """ Defines the action when the about item under help menu is clicked.
            Displays the basic description of News Search.
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("News Search is a GUI tool for searching relavant data in the Online Newspaper."+"\n\nDeveloped Using Python 3.5.2 and PyQt5 :D.\n")
        info = "10 lagwa do :P"
        msg.setInformativeText(info)
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def contribute(self):
        """ Opens my github XD
        """
        link = "https://github.com/addy007-icy"
        QDesktopServices.openUrl(QUrl(link))

    def openit(self, _links):
        """ The function which would open the link passed into it a variable
        """
        QDesktopServices.openUrl(QUrl(_links))

    def reconnect(self, signal, newhandler=None, oldhandler=None):
        """ The function which reconnects the Button to the link, 
            Erasing all previous connects to the button
        """
        while True:
            try:
                if oldhandler is not None:
                    signal.disconnect(oldhandler)
                else:
                    signal.disconnect()
            except TypeError:
                break
        if newhandler is not None:
            signal.connect(newhandler)

    def usage(self):
        """ HOW DO I USE THIS APP?
        """
        text = """
        <h4>In the given Query box type the document you want to look for</h4>
        <h4>Hit the Search button .. ..</h4>
        <h4>Watch the Engine Unfold.</h4>
        """
        details = QMessageBox()
        details.setText(text)
        details.setIcon(QMessageBox.Information)
        details.setWindowTitle("Usage")
        details.exec_()

    def modifyUI(self):
        """ The function which modifies the UI, takes input from query and gives output 
            and suggestions depending on the Query.
        """
        query = self.queryInput.text()
        out =  ""
        x = []
        x2, time = query_proc.more_results(query)
        # print (x)
        k = 1
        for a in x2:
            out = out + str(k) + a[0] + "\n\n"
            k = k + 1
        out = out +  " \n\nOutput Given in :" + str(time) + " seconds\n"
        self.SuggestionBrowser.setText(out)


        self.reconnect(self.buttonS1.clicked, lambda : self.openit(x2[0][2]))
        self.reconnect(self.buttonS2.clicked, lambda : self.openit(x2[1][2]))
        self.reconnect(self.buttonS3.clicked, lambda : self.openit(x2[2][2]))
        self.reconnect(self.buttonS4.clicked, lambda : self.openit(x2[3][2]))
        self.reconnect(self.buttonS5.clicked, lambda : self.openit(x2[4][2]))
        self.reconnect(self.buttonS6.clicked, lambda : self.openit(x2[5][2]))
        self.reconnect(self.buttonS7.clicked, lambda : self.openit(x2[6][2]))
        self.reconnect(self.buttonS8.clicked, lambda : self.openit(x2[7][2]))
        self.reconnect(self.buttonS9.clicked, lambda : self.openit(x2[8][2]))
        self.reconnect(self.buttonS10.clicked, lambda : self.openit(x2[9][2]))
        

        self.tab2layout.addWidget(self.buttonS1)
        self.tab2layout.addWidget(self.buttonS2)
        self.tab2layout.addWidget(self.buttonS3)
        self.tab2layout.addWidget(self.buttonS4)
        self.tab2layout.addWidget(self.buttonS5)
        self.tab2layout.addWidget(self.buttonS6)
        self.tab2layout.addWidget(self.buttonS7)
        self.tab2layout.addWidget(self.buttonS8)
        self.tab2layout.addWidget(self.buttonS9)
        self.tab2layout.addWidget(self.buttonS10)
        """
        Processsed Suggestions
        """

        x = []
        x, time = query_proc.ask_query(query)
        out = ""
        k = 1
        for a in x:
            out = out + str(k) + a[0] + "\n\n"
            k = k + 1
        out = out +  " \n\nOutput Given in :" + str(time) + " seconds\n"
        self.ResultBrowser.setText(out)

        self.reconnect(self.buttonP1.clicked, lambda : self.openit(x[0][2]))
        self.reconnect(self.buttonP2.clicked, lambda : self.openit(x[1][2]))
        self.reconnect(self.buttonP3.clicked, lambda : self.openit(x[2][2]))
        self.reconnect(self.buttonP4.clicked, lambda : self.openit(x[3][2]))
        self.reconnect(self.buttonP5.clicked, lambda : self.openit(x[4][2]))
        self.reconnect(self.buttonP6.clicked, lambda : self.openit(x[5][2]))
        self.reconnect(self.buttonP7.clicked, lambda : self.openit(x[6][2]))
        self.reconnect(self.buttonP8.clicked, lambda : self.openit(x[7][2]))
        self.reconnect(self.buttonP9.clicked, lambda : self.openit(x[8][2]))
        self.reconnect(self.buttonP10.clicked, lambda : self.openit(x[9][2]))
        

        self.tab1layout.addWidget(self.buttonP1)
        self.tab1layout.addWidget(self.buttonP2)
        self.tab1layout.addWidget(self.buttonP3)
        self.tab1layout.addWidget(self.buttonP4)
        self.tab1layout.addWidget(self.buttonP5)
        self.tab1layout.addWidget(self.buttonP6)
        self.tab1layout.addWidget(self.buttonP7)
        self.tab1layout.addWidget(self.buttonP8)
        self.tab1layout.addWidget(self.buttonP9)
        self.tab1layout.addWidget(self.buttonP10)



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
