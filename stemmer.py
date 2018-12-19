
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

    def is_cons(self, index):
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
                return not self.is_cons(index - 1)
        else:
            return True

    def count_cons(self):
        '''
        count the consonant in buffer[p0:j+1].
        '''
        cnt = 0
        i = self.p0
        while True:
            if i > self.j:
                return cnt
            if not(self.is_cons(i)):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return cnt
                if self.is_cons(i):
                    break
                i += 1
            i += 1
            cnt += 1
            while True:
                if i > self.j:
                    return cnt
                if not self.is_cons(i):
                    break
                i += 1
            i += 1

    def vowel_in_stem(self):
        '''
        if buffer[p0:j+1] contains a vowel
        '''
        for i in range(self.p0, self.j + 1):
            if not self.is_cons(i):
                return True
        return False

    def double_cons(self, index):
        '''
        if buffer[index-1:index+1] contain a double consonant
        '''
        if index < (self.p0 + 1):
            return True
        elif self.buffer[index] != self.buffer[index-1]:
            return False
        else:
            return self.is_cons(index)

    def end_with(self, string):
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

    def change_tail(self, string):
        '''
        change buffer[j+1,k+1] to string, update k
        '''
        lens = len(string)
        self.buffer = self.buffer[:self.j + 1] + string + self.buffer[self.j + lens + 1:]
        self.p = self.j + lens

    def replace(self, string):
        if self.count_cons()>0:
            self.change_tail(string)

    def step1(self):
        '''
        remove pluarals/ed/ing
        '''
        if self.end_with("sses"):
            self.change_tail("ss")
        elif self.end_with("ies"):
            self.change_tail("i")
        elif self.end_with("s") and not self.end_with("ss"):
            self.change_tail("")
        if self.end_with("eed"):
            self.replace("ee")
        elif (self.end_with("ed") or self.end_with("ing")) and self.vowel_in_stem():
            self.change_tail("")
            if self.end_with("at"):
                self.change_tail("ate")
            elif self.end_with("bl"):
                self.change_tail("ble")
            elif self.end_with("iz"):
                self.change_tail("ize")

    def step2(self):
        '''
        chage last y to i if there is another vowel in the stem.
        '''
        if self.end_with("y") and self.vowel_in_stem():
            self.buffer = self.buffer[:self.p] + 'i' + self.buffer[self.p + 1:]

    def step3(self):
        '''
        change double suffices to single ones.
        '''
        if self.end_with("ational"):
            self.replace("ate")
        elif self.end_with("tional"):
            self.replace("tion")
        elif self.end_with("enci"):
            self.replace("ence")
        elif self.end_with("anci"):
            self.replace("ance")
        elif self.end_with("izer"):
            self.replace("ize")
        elif self.end_with("bli"):
            self.replace("ble")
        elif self.end_with("alli"):
            self.replace("al")
        elif self.end_with("entli"):
            self.replace("ent")
        elif self.end_with("eli"):
            self.replace("e")
        elif self.end_with("ousli"):
            self.replace("ous")
        elif self.end_with("ization"):
            self.replace("ize")
        elif self.end_with("ation"):
            self.replace("ate")
        elif self.end_with("ator"):
            self.replace("ate")
        elif self.end_with("alism"):
            self.replace("al")
        elif self.end_with("iveness"):
            self.replace("ive")
        elif self.end_with("fulness"):
            self.replace("ful")
        elif self.end_with("ousness"):
            self.replace("ous")
        elif self.end_with("aliti"):
            self.replace("al")
        elif self.end_with("iviti"):
            self.replace("ive")
        elif self.end_with("biliti"):
            self.replace("ble")
        elif self.end_with("logi"):
            self.replace("log")

    def step4(self):
        '''
        ic,full，ness
        '''
        if self.end_with("icate"):
            self.replace("ic")
        elif self.end_with("ative"):
            self.replace("")
        elif self.end_with("alize"):
            self.replace("al")
        elif self.end_with("iciti"):
            self.replace("ic")
        elif self.end_with("ical"):
            self.replace("ic")
        elif self.end_with("ful"):
            self.replace("")
        elif self.end_with("ness"):
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
