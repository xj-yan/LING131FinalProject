
class Stemmer:
    '''
        Stemmer object, this is a simplified Porter stemmer,
        our stemmer is rule based, most of its functions replace suffix of a word to a new
        suffix base on several rules to change the word to its stem
    '''
    def __init__(self):
        self.buffer = ""  # buffer for word to be stemmed
        self.p0 = 0   # start of sterm
        self.p = 0    # end of the sterm
        self.j = 0   # general offset

    def isCons(self, index):
        '''
        if buffer[index] is a consonant.
        '''
        consonant_list = ['a', 'e', 'i', 'o', 'u']
        if self.buffer[index] in consonant_list:
            return False
        elif self.buffer[index] == 'y':
            if index == self.p0:
                return True
            else:
                return not self.isCons(index - 1)
        else:
            return True

    def countCons(self):
        '''
        count the consonant in buffer[p0:j+1].
        '''
        cnt = 0
        i = self.p0
        while True:
            if i > self.j:
                return cnt
            if not(self.isCons(i)):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return cnt
                if self.isCons(i):
                    break
                i += 1
            i += 1
            cnt += 1
            while True:
                if i > self.j:
                    return cnt
                if not self.isCons(i):
                    break
                i += 1
            i += 1

    def vowelInStem(self):
        '''
        if buffer[p0:j+1] contains a vowel
        '''
        for i in range(self.p0, self.j + 1):
            if not self.isCons(i):
                return True
        return False

    def doubleCons(self, index):
        '''
        if buffer[index-1:index+1] contain a double consonant
        '''
        if index < (self.p0 + 1):
            return True
        elif self.buffer[index] != self.buffer[index-1]:
            return False
        else:
            return self.isCons(index)

    def endWithStr(self, string):
        '''
        if buffer[p0:p+1] ends with string
        '''
        lens= len(string)
        if lens > (self.p - self.p0 + 1):
            return False
        elif self.buffer[self.p - lens + 1 : self.p + 1] != string:
            return False
        self.j = self.p - lens
        return True

    def changeTail(self,string):
        '''
        change buffer[j+1,k+1] to string, update k
        '''
        lens = len(string)
        self.buffer = self.buffer[:self.j + 1] + string + self.buffer[self.j + lens + 1:]
        self.p = self.j + lens

    def replace(self, string):
        if self.countCons()>0:
            self.changeTail(string)

    def step1(self):
        '''
        remove pluarals/ed/ing
        '''
        if self.endWithStr("sses"):
            self.changeTail("ss")
        elif self.endWithStr("ies"):
            self.changeTail("i")
        elif self.endWithStr("s") and not self.endWithStr("ss"):
            self.changeTail("")
        if self.endWithStr("eed"):
            self.replace("ee")
        elif (self.endWithStr("ed") or self.endWithStr("ing")) and self.vowelInStem():
            self.changeTail("")
            if self.endWithStr("at"):
                self.changeTail("ate")
            elif self.endWithStr("bl"):
                self.changeTail("ble")
            elif self.endWithStr("iz"):
                self.changeTail("ize")

    def step2(self):
        '''
        chage last y to i if there is another vowel in the stem.
        '''
        if self.endWithStr("y") and self.vowelInStem():
            self.buffer = self.buffer[:self.p] + 'i' + self.buffer[self.p + 1:]

    def step3(self):
        '''
        change double suffices to single ones.
        '''
        if self.endWithStr("ational"):
            self.replace("ate")
        elif self.endWithStr("tional"):
            self.replace("tion")
        elif self.endWithStr("enci"):
            self.replace("ence")
        elif self.endWithStr("anci"):
            self.replace("ance")
        elif self.endWithStr("izer"):
            self.replace("ize")
        elif self.endWithStr("bli"):
            self.replace("ble")
        elif self.endWithStr("alli"):
            self.replace("al")
        elif self.endWithStr("entli"):
            self.replace("ent")
        elif self.endWithStr("eli"):
            self.replace("e")
        elif self.endWithStr("ousli"):
            self.replace("ous")
        elif self.endWithStr("ization"):
            self.replace("ize")
        elif self.endWithStr("ation"):
            self.replace("ate")
        elif self.endWithStr("ator"):
            self.replace("ate")
        elif self.endWithStr("alism"):
            self.replace("al")
        elif self.endWithStr("iveness"):
            self.replace("ive")
        elif self.endWithStr("fulness"):
            self.replace("ful")
        elif self.endWithStr("ousness"):
            self.replace("ous")
        elif self.endWithStr("aliti"):
            self.replace("al")
        elif self.endWithStr("iviti"):
            self.replace("ive")
        elif self.endWithStr("biliti"):
            self.replace("ble")
        elif self.endWithStr("logi"):
            self.replace("log")

    def step4(self):
        '''
        ic,full，ness
        '''
        if self.endWithStr("icate"):
            self.replace("ic")
        elif self.endWithStr("ative"):
            self.replace("")
        elif self.endWithStr("alize"):
            self.replace("al")
        elif self.endWithStr("iciti"):
            self.replace("ic")
        elif self.endWithStr("ical"):
            self.replace("ic")
        elif self.endWithStr("ful"):
            self.replace("")
        elif self.endWithStr("ness"):
            self.replace("")

    def stem(self, word):
        self.buffer = word
        self.p = len(word)-1
        self.p0 = 0
        self.step1()
        self.step2()
        self.step3()
        self.step4()
        return self.buffer[self.p0:self.p+1]